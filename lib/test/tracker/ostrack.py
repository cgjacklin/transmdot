import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import time

class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.z_dict2 = {}

        self.z_dict_list = []   # 模板列表



    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,   # 对(720,1280,3)的image进行裁剪
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    
    #   多机初始化
    def multi_initialize(self, image_a, image_b, init_info_a, init_info_b):
        
        # out_result = []

        # for image, info in enumerate(zip(img_list, info_list)):

            # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image_a, init_info_a['init_bbox'], self.params.template_factor,   # 对(720,1280,3)的image进行裁剪
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template


        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:          # 候选消除模块分别插入Vit的第几层
            template_bbox = self.transform_bbox_to_crop(init_info_a['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)


        # B机模板
        z_patch_arr2, resize_factor2, z_amask_arr2 = sample_target(image_b, init_info_b['init_bbox'], self.params.template_factor,   # 对(720,1280,3)的image进行裁剪
                                                    output_sz=self.params.template_size)
        self.z_patch_arr2 = z_patch_arr2
        template2 = self.preprocessor.process(z_patch_arr2, z_amask_arr2)
        with torch.no_grad():
            self.z_dict2 = template2

        self.box_mask_z2 = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:          # 候选消除模块分别插入Vit的第几层
            template_bbox2 = self.transform_bbox_to_crop(init_info_b['init_bbox'], resize_factor2,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z2 = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox2)


        # save states
        self.state = init_info_a['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = init_info_a['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            # out_result.append(all_boxes_save)
            return {"all_boxes": all_boxes_save}




    # 融合多机结果，response map取置信度最高的那个
    def Fusetrack(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定搜索区域
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        response_APCE = self.calAPCE(response)                # 计算平均峰值能量APCE

        pred_boxes, max_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}, max_score, response_APCE


    # 搜索A的搜索区域，融合AB模板，主要用这个
    def multi_Fusetrack(self, image_a, image_b, drone_id, info_a: dict = None, info_b: dict = None):          # self.state是上帧追踪结果，就是一个框
        H, W, _ = image_a.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(                  # 在forward这里要把两个模板传进去
                template=self.z_dict1.tensors, template2=self.z_dict2.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
                #template=self.z_dict1.tensors, template2=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)      # 无多模板

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        response_APCE = self.calAPCE(response)                # 计算平均峰值能量APCE

        pred_boxes, max_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_a, info_a['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking'+drone_id)

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region'+drone_id)
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template'+drone_id)
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map'+drone_id)
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann'+drone_id)

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search'+drone_id)

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}, max_score, response_APCE


    # 将B机搜索区域在A机中查询，然后重检测
    def search_redetect(self, image_a, image_b, drone_id, state_b, tmp_factor = 4.0, tmp_s_factor = 12.0, info_a: dict = None, info_b: dict = None):       

        print(drone_id, "机丢失， cross redetect")

        H, W, _ = image_a.shape
        #self.frame_id += 1
        
        

        x_patch_arr_b, resize_factor_b, x_amask_arr_b = sample_target(image_b, state_b, tmp_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)     # 将B的搜索区域裁出来


        search_b = self.preprocessor.process(x_patch_arr_b, x_amask_arr_b)

        #print("before: ", self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, tmp_s_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域

        search_a = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            #x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(                  # 在forward这里要把两个模板传进去
                template=search_b.tensors, template2=search_b.tensors, search=search_a.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        #response = self.output_window * pred_score_map

        #response_APCE = self.calAPCE(response)                # 计算平均峰值能量APCE

        pred_boxes, max_score = self.network.box_head.cal_bbox(pred_score_map, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score
        # print("pred_boxes: ", pred_boxes)
        # print("max score: ", max_score)



        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        #print("after: ", self.state)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_a, info_a['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking'+drone_id)

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region'+drone_id)
                self.visdom.register(torch.from_numpy(x_patch_arr_b).permute(2, 0, 1), 'image', 1, 'template'+drone_id)
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map'+drone_id)
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann'+drone_id)

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search'+drone_id)

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        self.frame_id -= 1
        out_t, max_score_t, response_APCE_t = self.multi_Fusetrack(image_a, image_b, drone_id, info_a, info_b)


        tmp_state = self.state.copy()         # 保存中间状态



        ############################# 在center中框个Box ##################################
        if tmp_s_factor > 8:
            center_state = tmp_state.copy()
            center_state[1] , center_state[0] = H/2 , W/2
            x_patch_arr_ctr, resize_factor_ctr, x_amask_arr_ctr = sample_target(image_a, center_state, tmp_s_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
            search_ctr = self.preprocessor.process(x_patch_arr_ctr, x_amask_arr_ctr)

            # with torch.no_grad():
            #     out_dict_ctr_z = self.network.forward(
            #         template=self.z_dict1.tensors, template2=self.z_dict2.tensors, search=search_ctr.tensors, ce_template_mask=self.box_mask_z)   # 在中间用模板检测

            # pred_score_map_ctr_z = out_dict_ctr_z['score_map']
            # pred_boxes_ctr_z, max_score_ctr_z = self.network.box_head.cal_bbox(pred_score_map_ctr_z, out_dict_ctr_z['size_map'], out_dict_ctr_z['offset_map'], return_score=True)   # 获得最大score

            with torch.no_grad():
                out_dict_ctr = self.network.forward(                 
                    template=search_b.tensors, template2=search_b.tensors, search=search_ctr.tensors, ce_template_mask=self.box_mask_z)   # 在中间用搜索区域检测
            pred_score_map_ctr = out_dict_ctr['score_map']
            pred_boxes_ctr, max_score_ctr = self.network.box_head.cal_bbox(pred_score_map_ctr, out_dict_ctr['size_map'], out_dict_ctr['offset_map'], return_score=True)   # 获得最大score

            

            if (max_score_ctr - max_score) > 0.1:
                print("使用了中心框")
                print(max_score_ctr)
                print(max_score)
                # if (max_score_ctr - max_score_ctr_z) > 0:

                pred_boxes_ctr = pred_boxes_ctr.view(-1, 4)
                pred_box_ctr = (pred_boxes_ctr.mean(
                    dim=0) * self.params.search_size / resize_factor_ctr).tolist()  # (cx, cy, w, h) [0,1]
                self.state = clip_box(self.map_box_back(pred_box_ctr, resize_factor_ctr), H, W, margin=10)

                self.frame_id -= 1
                out_ctr, max_score_ctr, response_APCE_ctr = self.multi_Fusetrack(image_a, image_b, drone_id, info_a, info_b)


                if max_score_ctr > max_score_t:
                    return out_ctr, max_score_ctr, response_APCE_ctr
                else:
                    self.state = tmp_state
    

        ##################################################################################


        # if self.save_all_boxes:
        #     '''save all predictions'''
        #     all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        #     all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
        #     return {"target_bbox": self.state,
        #             "all_boxes": all_boxes_save}
        # else:
        return out_t, max_score_t, response_APCE_t


# general 重检测
    def general_redetect(self, image_a, image_b, drone_id, tmp_s_factor = 7.0, info_a: dict = None, info_b: dict = None):       

        print(drone_id, "机丢失， general redetect")

        H, W, _ = image_a.shape
        #self.frame_id += 1
        

        #print("before: ", self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, tmp_s_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域

        search_a = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            #x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(                  # 在forward这里要把两个模板传进去
                template=self.z_dict1.tensors, template2=self.z_dict2.tensors, search=search_a.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        #response = self.output_window * pred_score_map

        #response_APCE = self.calAPCE(response)                # 计算平均峰值能量APCE

        pred_boxes, max_score = self.network.box_head.cal_bbox(pred_score_map, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score


        ############################# 在center中框个Box ##################################
        center_state = self.state
        center_state[1] , center_state[0] = H/2 , W/2
        x_patch_arr_ctr, resize_factor_ctr, x_amask_arr_ctr = sample_target(image_a, center_state, tmp_s_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        search_ctr = self.preprocessor.process(x_patch_arr_ctr, x_amask_arr_ctr)
        with torch.no_grad():
            out_dict_ctr = self.network.forward(                  # 在forward这里要把两个模板传进去
                template=self.z_dict1.tensors, template2=self.z_dict2.tensors, search=search_ctr.tensors, ce_template_mask=self.box_mask_z)
        pred_score_map_ctr = out_dict_ctr['score_map']
        pred_boxes_ctr, max_score_ctr = self.network.box_head.cal_bbox(pred_score_map_ctr, out_dict_ctr['size_map'], out_dict_ctr['offset_map'], return_score=True)   # 获得最大score

        if (max_score_ctr - max_score) > 0:
            pred_boxes = pred_boxes_ctr
            max_score = max_score_ctr
        ##################################################################################

        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        #print("after: ", self.state)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_a, info_a['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking'+drone_id)

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region'+drone_id)
                self.visdom.register(torch.from_numpy(x_patch_arr_ctr).permute(2, 0, 1), 'image', 1, 'search_region'+drone_id+'center')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template'+drone_id)
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map'+drone_id)
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann'+drone_id)

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search'+drone_id)

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        self.frame_id -= 1
        out, max_score, response_APCE = self.multi_Fusetrack(image_a, image_b, drone_id, info_a, info_b)




        # if self.save_all_boxes:
        #     '''save all predictions'''
        #     all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        #     all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
        #     return {"target_bbox": self.state,
        #             "all_boxes": all_boxes_save}
        # else:
        return out, max_score, response_APCE


    # 搜索A的搜索区域，融合AB模板, 和AB搜索区域
    def multi_Fusetrack2(self, image_a, image_b, state2, info_a: dict = None, info_b: dict = None):
        H, W, _ = image_a.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        x_patch_arr2, resize_factor2, x_amask_arr2 = sample_target(image_b, state2, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定B的搜索区域
        search2 = self.preprocessor.process(x_patch_arr2, x_amask_arr2)
        
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(                  # 在forward这里要把两个模板传进去，和两个搜索区域
                template=self.z_dict1.tensors, template2=self.z_dict2.tensors, search=x_dict.tensors, search2=search2.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, max_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_a, info_a['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}, max_score

    
    # 计算平均峰值相关能量
    def calAPCE(self, response):
        faltted_response = response.flatten(1)
        max_score, idx_max = torch.max(faltted_response, dim=1, keepdim=True)
        min_score, idx_min = torch.min(faltted_response, dim=1, keepdim=True)

        _, response_len = faltted_response.shape

        tmp_sum = 0
        for i, score in enumerate(faltted_response.squeeze()):         # squeeze是把维度为1的去掉
            tmp_sum += (score - min_score) ** 2

        bottom = tmp_sum / (i+1)

        APEC = ((max_score - min_score) ** 2) / bottom

        # print("APCE: ", APEC)

        return APEC

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
