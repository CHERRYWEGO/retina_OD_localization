import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pose.utils.evaluation import get_preds
import numpy as np

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range * 255

class EyeMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.dis_metric = lambda x, y: np.sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2))
        self.bilinear = None
        self.mesh_x = None
        self.mesh_refresh = False
        self.result = []

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

        self.real_sum = 0.0
        self.real_sum_box = 0.0
        self.real_sum_oval = 0.0
        self.real_sum_dis8 = 0.0
        self.real_sum_dis4 = 0.0
        self.real_sum_dis2 = 0.0
        self.real_sum_dis1 = 0.0
        self.real_sum_dis = 0.0
        self.real_sum_iou = 0.0
        self.real_sum_od = 0.0
        self.real_sum_fovea = 0.0

        self.real_avg = 0.0
        self.real_avg_box = 0.0
        self.real_avg_oval = 0.0
        self.real_avg_dis8 = 0.0
        self.real_avg_dis4 = 0.0
        self.real_avg_dis2 = 0.0
        self.real_avg_dis1 = 0.0
        self.real_avg_dis = 0.0
        self.real_avg_iou = 0.0
        self.real_avg_od = 0.0
        self.real_avg_fovea = 0.0


    def get_preds(self, preds, ratio=0.7):
        all_preds = []
        # preds = preds.tolist()
        self.line_len = preds.shape[3]
        # if self.mesh_x is None or self.mesh_refresh:
        a = [i for i in range(self.line_len)]
        self.mesh_x, self.mesh_y = np.meshgrid(a, a)
        self.mesh_x = np.array(self.mesh_x)
        self.mesh_y = np.array(self.mesh_y)
        for i in range(0, len(preds)):
            try:
                max = [torch.max(preds[i][0]).tolist()]

                mask = np.array(preds[i][0].tolist()) > (max[0] * ratio if max[0] > 0 else max[0] / ratio)
                p1 = [int(round(self.mesh_x[mask].mean())), int(round(self.mesh_y[mask].mean()))]

                all_preds.append(p1)
                # max = [torch.max(preds[i][0]).tolist(), torch.max(preds[i][1]).tolist()]
                #
                # mask = np.array(preds[i][0].tolist()) > (max[0] * ratio if max[0] > 0 else max[0] / ratio)
                # p1 = [int(self.mesh_x[mask].mean()), int(self.mesh_y[mask].mean())]
                #
                #
                # mask = np.array(preds[i][1].tolist()) > (max[1] * ratio if max[1] > 0 else max[1] / ratio)
                # p2 = [int(self.mesh_x[mask].mean()), int(self.mesh_y[mask].mean())]
                # all_preds.append([p1, p2])
            except:
                pass
            # all_preds.append([[max_pos[0] % line_len, max_pos[0] // line_len],
            #                   [max_pos[1] % line_len, max_pos[1] // line_len]])
        return all_preds

    def update_all(self, inputs_tensor, meta, target=None, bilinear_test=False):

        # inputs = np.array(self.get_preds(inputs_tensor))
        # n_target = meta['npts'].tolist()
        # n_target = (meta['pts_box'].clone().float() * meta['ratio']).long().tolist()

        # for i in range(len(inputs)):
        #     dis_i = inputs[i]
        #     dis_t = n_target[i]
        #     if dis_i[0]>dis_t[0] and dis_i[0]<dis_t[2] and dis_i[1]>dis_t[1] and dis_i[1]<dis_t[3] :
        #         self.sum += 1
        #     self.count += 1
        # self. avg = self.sum / self.count

        if bilinear_test:
            target = meta['pts_box'].tolist()

            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[0][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            inputs = np.array(self.get_preds(inputs_origin))

            for i in range(len(inputs)):
                # dis_i = inputs[i] / ratios[i]
                dis_i = inputs[i]
                dis_t = target[i]

                self.count += 1
                # box
                if dis_i[0] > dis_t[0] and dis_i[0] < dis_t[2] and dis_i[1] > dis_t[1] and dis_i[1] < dis_t[3]:
                    self.real_sum_box += 1
                # else:
                    img_path = meta['img_path'][i]
                    img_name = img_path.split("/")[-1]
                    img_res = cv2.imread(img_path)
                    cv2.line(img_res, (dis_i[0] - 10, dis_i[1]), (dis_i[0] + 10, dis_i[1]), (0, 0, 0), 2)
                    cv2.line(img_res, (dis_i[0], dis_i[1] - 10), (dis_i[0], dis_i[1] + 10), (0, 0, 0 ), 2)
                    # cv2.rectangle(img_res, (dis_t[0], dis_t[1]), (dis_t[2], dis_t[3]), (0, 255, 0), 1)
                    # cv2.circle(img_res, (dis_i[0], dis_i[1]), 2, (0, 255, 0), -1)
                    cv2.imwrite('res/'+img_name, img_res)

                # oval
                oval_mask = np.zeros((256, 256, 3), np.uint8)
                ct_x = int(round((dis_t[0] + dis_t[2]) / 2))
                ct_y = int(round((dis_t[1] + dis_t[3]) / 2))
                lside = int(round(max(dis_t[2] - dis_t[0], dis_t[3] - dis_t[1])/2))
                sside = int(round(min(dis_t[2] - dis_t[0], dis_t[3] - dis_t[1])/2))
                cv2.ellipse(oval_mask, (ct_x, ct_y), (lside, sside), 0, 0, 360, (255, 255, 255), -1)  # 画椭圆

                if oval_mask[dis_i[1]][dis_i[0]][0] == 255:
                    self.real_sum_oval += 1

                #r
                optic_r = lside
                dis_err = self.dis_metric(dis_i, (ct_x, ct_y))

                # if oval_mask[dis_i[1]][dis_i[0]][0] == 255:
                self.real_sum_dis += dis_err
                if (dis_err < optic_r / 8):    self.real_sum_dis8 += 1
                if (dis_err < optic_r / 4):    self.real_sum_dis4 += 1
                if (dis_err < optic_r / 2):    self.real_sum_dis2 += 1
                if (dis_err < optic_r / 1):    self.real_sum_dis1 += 1


            self.real_avg_box = self.real_sum_box / self.count
            self.real_avg_oval = self.real_sum_oval / self.count
            self.real_avg_dis = self.real_sum_dis / self.count
            self.real_avg_dis8 = self.real_sum_dis8 / self.count
            self.real_avg_dis4 = self.real_sum_dis4 / self.count
            self.real_avg_dis2 = self.real_sum_dis2 / self.count
            self.real_avg_dis1 = self.real_sum_dis1 / self.count
            # input = get_preds(inputs, False)

    def update_box(self, inputs_tensor, meta, target=None, bilinear_test=False):

        # inputs = np.array(self.get_preds(inputs_tensor))
        # # n_target = meta['npts'].tolist()
        # n_target = (meta['pts_box'].clone().float() * meta['ratio']).long().tolist()
        #
        # for i in range(len(inputs)):
        #     dis_i = inputs[i]
        #     dis_t = n_target[i]
        #     if dis_i[0]>dis_t[0] and dis_i[0]<dis_t[2] and dis_i[1]>dis_t[1] and dis_i[1]<dis_t[3] :
        #         self.sum += 1
        #     self.count += 1
        # self. avg = self.sum / self.count

        if bilinear_test:
            target = meta['pts_box'].tolist()
            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[0][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            inputs = np.array(self.get_preds(inputs_origin))
            for i in range(len(inputs)):
                # dis_i = inputs[i] / ratios[i]
                dis_i = inputs[i]
                dis_t = target[i]

                # self.real_sum += self.dis_mertric(dis_i, dis_t)
                self.count += 1
                if dis_i[0] > dis_t[0] and dis_i[0] < dis_t[2] and dis_i[1] > dis_t[1] and dis_i[1] < dis_t[3]:
                    self.real_sum += 1
                # else:
                #     img_path = meta['img_path'][i]
                #     img_name = img_path.split("/")[-1]
                #     img_res = cv2.imread(img_path)
                #     cv2.rectangle(img_res, (dis_t[0], dis_t[1]), (dis_t[2], dis_t[3]), (0, 255, 0), 1)
                #     cv2.circle(img_res, (dis_i[0], dis_i[1]), 2, (0, 255, 0), -1)
                #     cv2.imwrite('fault/'+img_name, img_res)
                    

            self.real_avg = self.real_sum / self.count

    def update_circle(self, inputs_tensor, meta, bilinear_test=False):
        # inputs = np.array(self.get_preds(inputs_tensor))
        target = meta['mask_path']

        if bilinear_test:
            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[0][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            inputs = np.array(self.get_preds(inputs_origin))
            for i in range(len(inputs)):
                # dis_i = inputs[i] / ratios[i]
                dis_i = inputs[i]
                mask_t = plt.imread(target[i])

                self.count += 1
                # self.real_sum += self.dis_mertric(dis_i, dis_t)
                if mask_t[dis_i[1]][dis_i[0]] > 128:
                    self.real_sum += 1
                # else:
                #     img_path = meta['img_path'][i]
                #     img_name = img_path.split("/")[-1]
                #     img_res = cv2.imread(img_path)
                #     cv2.circle(img_res, (dis_i[0], dis_i[1]), 2, (0, 255, 0), -1)
                #     cv2.imwrite('fault/'+img_name, img_res)
                    #
                    #
                    # heat_map = inputs_tensor[i].cpu().numpy()
                    # img = imgs[i]
                    # img = img.transpose(1, 2, 0)
                    #
                    # heat_mask_tmap = np.sum(heat_map, 0)
                    #
                    # heat_map = heat_map.reshape(*heat_map.shape, -1)
                    # zeros = np.zeros([*heat_map.shape])
                    # heat_map = np.concatenate([zeros, heat_map, zeros], axis=2)
                    #
                    # heat_map = list(resize(heat_map, (512, 512)))
                    # heat_map = np.array(heat_map)
                    #
                    # fimg = img * 1 + heat_map * 1.5
                    # minVal = np.amin(fimg)
                    # maxVal = np.amax(fimg)
                    # outufuncXArray = normalize_func(minVal, maxVal, 0, 1)(fimg)  # the result is a ufunc object
                    # dataXArray = outufuncXArray.astype(float)  # cast ufunc object ndarray to float ndarray
                    # plt.imsave('fault/fimg_{}'.format(int(self.count)), dataXArray)

                #     np.save('imgs_{}'.format(int(self.count)), imgs[i])
                #     heat_map = inputs_tensor[i].cpu().numpy()
                #     np.save('heat_map_{}'.format(int(self.count)), heat_map)

                # self.real_sum += self.dis_mertric(dis_i[1], dis_t[1])

                # print(self.dis_mertric(dis_i[0], dis_t[0]))
                # print(self.dis_mertric(dis_i[1], dis_t[1]))
                # print()

                # with open('data.txt', 'w') as f:
                #     f.write(str(i))
                #     f.write('\n')
                #     f.write(str(self.dis_mertric(dis_i[0], dis_t[0])))
                #     f.write('\n')
                #     f.write(str(self.dis_mertric(dis_i[1], dis_t[1])))
                #     f.write('\n')
                #     f.write('\n')

                # self.count += 2
            self.real_avg = self.real_sum / self.count
            # input = get_preds(inputs, False)

    def update_iou(self, inputs_tensor, meta, target=None, bilinear_test=False):

        inputs = np.array(self.get_preds(inputs_tensor))
        # n_target = meta['npts'].tolist()
        n_target = (meta['pts_box'].clone().float() * meta['ratio']).long().tolist()

        for i in range(len(inputs)):
            dis_i = inputs[i]
            dis_t = n_target[i]
            if dis_i[0]>dis_t[0] and dis_i[0]<dis_t[2] and dis_i[1]>dis_t[1] and dis_i[1]<dis_t[3] :
                self.sum += 1
            self.count += 1
        self. avg = self.sum / self.count

        if bilinear_test:
            target = meta['pts_box'].tolist()
            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[i][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            inputs = np.array(self.get_preds(inputs_origin))
            for i in range(len(inputs)):
                # dis_i = inputs[i] / ratios[i]
                dis_i = inputs[i]
                dis_t = target[i]

                # self.real_sum += self.dis_mertric(dis_i, dis_t)
                # self.count += 1
                if dis_i[0] > dis_t[0] and dis_i[0] < dis_t[2] and dis_i[1] > dis_t[1] and dis_i[1] < dis_t[3]:
                    self.real_sum += 1
                # else:
                #     img_path = meta['img_path'][i]
                #     img_name = img_path.split("/")[-1]
                #     img_res = cv2.imread(img_path)
                #     cv2.rectangle(img_res, (dis_t[0], dis_t[1]), (dis_t[2], dis_t[3]), (0, 255, 0), 1)
                #     cv2.circle(img_res, (dis_i[0], dis_i[1]), 2, (0, 255, 0), -1)
                #     cv2.imwrite('fault/'+img_name, img_res)
                    #
                    #
                    # heat_map = inputs_tensor[i].cpu().numpy()
                    # img = imgs[i]
                    # img = img.transpose(1, 2, 0)
                    #
                    # heat_map = np.sum(heat_map, 0)
                    #
                    # heat_map = heat_map.reshape(*heat_map.shape, -1)
                    # zeros = np.zeros([*heat_map.shape])
                    # heat_map = np.concatenate([zeros, heat_map, zeros], axis=2)
                    #
                    # heat_map = list(resize(heat_map, (512, 512)))
                    # heat_map = np.array(heat_map)
                    #
                    # fimg = img * 1 + heat_map * 1.5
                    # minVal = np.amin(fimg)
                    # maxVal = np.amax(fimg)
                    # outufuncXArray = normalize_func(minVal, maxVal, 0, 1)(fimg)  # the result is a ufunc object
                    # dataXArray = outufuncXArray.astype(float)  # cast ufunc object ndarray to float ndarray
                    # plt.imsave('fault/fimg_{}'.format(int(self.count)), dataXArray)

                #     np.save('imgs_{}'.format(int(self.count)), imgs[i])
                #     heat_map = inputs_tensor[i].cpu().numpy()
                #     np.save('heat_map_{}'.format(int(self.count)), heat_map)

                # self.real_sum += self.dis_mertric(dis_i[1], dis_t[1])

                # print(self.dis_mertric(dis_i[0], dis_t[0]))
                # print(self.dis_mertric(dis_i[1], dis_t[1]))
                # print()

                # with open('data.txt', 'w') as f:
                #     f.write(str(i))
                #     f.write('\n')
                #     f.write(str(self.dis_mertric(dis_i[0], dis_t[0])))
                #     f.write('\n')
                #     f.write(str(self.dis_mertric(dis_i[1], dis_t[1])))
                #     f.write('\n')
                #     f.write('\n')

                # self.count += 2
            self.real_avg = self.real_sum / self.count
            # input = get_preds(inputs, False)

    def update_heatmap(self, inputs_tensor, meta, target=None, bilinear_test=False):
        # target = meta['mask_path']
        n_target = target.cpu().numpy()
        n_inputs = inputs_tensor.cpu().numpy()
        for i in range(len(n_inputs)):
            dis_t = n_target[i]
            dis_i = n_inputs[i]
            heat_map = np.sum(dis_i, 0)
            heat_map = normalization(heat_map)
            heat_map = heat_map.astype(np.uint8)

            mask_h = np.array(heat_map) > 127
            mask_t = np.array(dis_t) > 0
            mask = mask_h + mask_t
            inter_mask = mask_h * mask_t
            iou = np.sum(inter_mask) / np.sum(mask)

            self.count += 1
            self.real_sum_iou += iou

        self.real_avg_iou = self.real_sum_iou / self.count

        if bilinear_test:
            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[0][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            dis_i = inputs_origin.cpu().numpy()

            for i in range(len(inputs_origin)):
                heat_map = dis_i[i]
                heat_map = np.sum(heat_map, 0)
                heat_map = normalization(heat_map)
                heat_map = heat_map.astype(np.uint8)

                # mask_t = cv2.imread(target[i])
                # mask_t = cv2.cvtColor(mask_t, cv2.COLOR_BGR2GRAY)

                img_path = meta['img_path'][i]
                img_name = img_path.split("/")[-1]
                cv2.imwrite('heatmap/' + img_name, heat_map)

    def update_res(self, inputs_tensor, meta, target=None, bilinear_test=False):

        if bilinear_test:
            # target = meta['pts_box'].tolist()

            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[0][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            inputs = np.array(self.get_preds(inputs_origin))

            for i in range(len(inputs)):
                # dis_i = inputs[i] / ratios[i]
                dis_i = inputs[i]
                
                img_path = meta['img_path'][i]
                img_name = img_path.split("/")[-1]
                img_res = cv2.imread(img_path)
                cv2.line(img_res, (dis_i[0] - 10, dis_i[1]), (dis_i[0] + 10, dis_i[1]), (0, 0, 0), 2)
                cv2.line(img_res, (dis_i[0], dis_i[1] - 10), (dis_i[0], dis_i[1] + 10), (0, 0, 0 ), 2)
                cv2.imwrite('OD2019/'+img_name, img_res)

                self.result.append([img_name, dis_i[0], dis_i[1]])

            # input = get_preds(inputs, False)
