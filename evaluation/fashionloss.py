import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pose.utils.evaluation import get_preds
import numpy as np

# normalize ndarray
# def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1 ):
#     def normalizeFunc(x):
#         r=(x-minVal)*newMaxValue/(maxVal-minVal) + newMinValue
#         return r
#     return np.frompyfunc(normalizeFunc, 1, 1)
#
# def transfer(image):
#
#     data = image.copy()
#     min_max = []  # range of channls
#     h, w, c = data.shape
#     for i in range(c):
#         min_d, max_d = (np.min(data[:, :, i]), np.max(data[:, :, i]))
#         min_max.append((min_d, max_d))
#         zone = max_d - min_d
#         if zone < 0.1:
#             zone = 0.1
#         data[:, :, i] = 1.0 * (data[:, :, i] - min_d) / zone * 255
#     data = data.astype('uint8')
#     return data, min_max
#
# def re_transfer(image, min_max):
#     data = image.copy()
#     data = data.astype('float16')
#     h, w, c = data.shape
#     for i in range(c):
#         min_d, max_d = min_max[i]
#         min_now, max_now = (np.min(data[:, :, i]), np.max(data[:, :, i]))
#         zone = (max_now - min_now)
#         if zone < 0.1:
#             zone = 0.1
#
#         data[:, :, i] = 1.0 * (data[:, :, i] - min_now) / zone * (max_d - min_d) + min_d
#
#     return data
#
# def resize(data,sz):
#
#     image, min_max = transfer(data)
#     image = Image.fromarray(image)
#     image = image.resize(sz)
#     image = np.array(image)
#     image = re_transfer(image, min_max)
#     return image
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range * 255

def heatmapcenter(heatmaps, numclass):
    result = []
    scale = heatmaps[0][0].shape[0]
    for j in range(0, len(heatmaps)):
        points = []
        for i in range(0, numclass):
            total = heatmaps[j][i]
            r = 1

            max_heat = -1000
            width = 0
            height = 0

            for X in range(r, scale - r):
                for Y in range(r, scale - r):
                    currentHeat = 0
                    for k in range(X - r, X + ( r + 1 )):
                        for h in range(Y - r, Y + ( r + 1 )):
                            currentHeat += total[h][k]
                    if currentHeat > max_heat:
                        max_heat = currentHeat
                        (width, height) = (k, h)
            points.append((width, height))
        result.append(points)
    return result

class FashionMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.stand_point = {
            'blouse': [5, 6],
            'skirt': [15, 16],
            'outwear': [5, 6],
            'dress': [5, 6],
            'trousers': [15, 16]
        }
        self.reset()

        self.class_points = {
            'blouse': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
            'skirt': [15, 16, 17, 18],
            'outwear': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'dress': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
            'trousers': [15, 16, 19, 20, 21, 22, 23]
        }

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0


    def update(self, inputs, targets, meta, num_class):

        target = meta['pts']
        input = get_preds(inputs, False)
        result_dis = 0
        result_point = 0
        # target_gap = 3
        # gap = 3
        covers = meta['cover']
        kinds = meta['kind']
        ratio = 4
        for j in range(0, len(input)):
            good_point = 0
            total_dis = 0
            sp = self.stand_point[kinds[j]]
            # cp = self.class_points[kinds[j]]
            st_point1, st_point2 = torch.FloatTensor(
                [target[j][sp[0]][0], target[j][sp[0]][1]]), torch.FloatTensor(
                [target[j][sp[1]][0], target[j][sp[1]][1]])
            stand_dis = torch.sqrt(torch.FloatTensor([torch.sum(torch.pow(st_point1 - st_point2, 2))]))
            for i in range(0, num_class):
                if covers[j][i] != 1:
                    continue
                point1, point2 = torch.FloatTensor([target[j][i][0], target[j][i][1]]), torch.FloatTensor(
                    [input[j][i][0] * ratio, input[j][i][1] * ratio])
                total_dis += torch.sqrt(torch.FloatTensor([torch.sum(torch.pow(point1 - point2, 2))]))
                good_point += 1

            if stand_dis[0]> 1 and good_point > 0:
                result_dis += torch.div(total_dis, stand_dis)[0]
                result_point += good_point
        if result_point < 1 or result_dis < 0.001:
            print("extremely wrong")
        else:
            self.val = result_dis
            self.avg = (self.avg * self.count + self.val) / (self.count + result_point)
            self.val = self.val / result_point
            self.count += result_point

class EyeMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.dis_metric = lambda x, y: np.sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2))
        self.bilinear = None
        self.mesh_x = None
        self.mesh_refresh = False

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


    def update(self, inputs_tensor, meta, target=None, bilinear_test=False):

        # inputs = np.array(self.get_preds(inputs_tensor))
        # n_target = meta['npts'].tolist()
        # # n_target = meta['pts_box']
        # for i in range(len(inputs)):
        #     dis_i = inputs[i]
        #     dis_t = n_target[i]
        #     self.sum += self.dis_metric(dis_i, dis_t)
        #     # self.sum += self.dis_metric(dis_i[1], dis_t[1])
        #     self.count += 1
        # self.avg = self.sum / self.count
        if bilinear_test:
            target = meta['pts'].tolist()
            if self.bilinear is None:
                ratios = meta['ratio'].tolist()
                self.bilinear = torch.nn.Upsample(scale_factor=int(1 / ratios[0][0]), mode='bilinear')
            inputs_origin = self.bilinear(inputs_tensor)
            inputs = np.array(self.get_preds(inputs_origin))
            for i in range(len(inputs)):
                self.count += 1
                # dis_i = inputs[i] / ratios[i]
                dis_i = inputs[i]
                dis_t = target[i]

                self.real_sum_od += self.dis_metric(dis_i[0], dis_t[0])
                self.real_sum_fovea += self.dis_metric(dis_i[1], dis_t[1])

                # img_path = meta['img_path'][i]
                # img_name = img_path.split("/")[-1]
                # img_res = cv2.imread(img_path)
                # cv2.circle(img_res, (int(dis_t[0]), int(dis_t[1])), 2, (0, 255, 0), -1)
                # cv2.circle(img_res, (dis_i[0], dis_i[1]), 2, (255, 0, 0), -1)
                # cv2.imwrite('res/' + img_name, img_res)
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
            self.real_avg_od = self.real_sum_od / self.count
            self.real_avg_fovea = self.real_sum_fovea / self.count
            self.real_avg = (self.real_avg_od + self.real_avg_fovea) / 2
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
