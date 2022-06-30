# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

# own imports
from os import listdir, mkdir
from os.path import isfile,join, isdir

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadImagesImageList
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

import numpy as np

from os.path import basename,splitext


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        is_image_list=False,  # if is_image == True, then let source be an array of cv2/numpy images
        question_nums=None,  # added this myself, add a list with question numbers of the same length as the source list
        view_img=False,  # show results
        view_crop=False,  # added this myself, displays crops
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        return_coordlist=False,  # added this myself, returns list of cropped images per image
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        webcam=False,
        return_image=False  # returns full image with annotation
):
    if is_image_list:
        save_img = not nosave
    else:

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # copy question num
    question_nums_found = []

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    elif is_image_list:
        dataset = LoadImagesImageList(question_nums, source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    list_cropped_images = []

    if is_image_list:

        for question_num, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im, augment=augment, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):


                        if save_img or save_crop or view_img or view_crop or return_coordlist or return_image:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                            if return_image:
                                annotator.box_label(xyxy, label, color=colors(c, True)) # hide this, because it gives a red line
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{question_num}.jpg', BGR=True)
                        if view_crop or return_coordlist:
                            cropped_image = save_one_box(xyxy, imc,
                                                         file=save_dir / 'crops' / names[c] / f'{question_num}.jpg',
                                                         BGR=True, save=False)
                            if view_crop:
                                cv2.imshow(str(conf), cropped_image)
                                cv2.waitKey(0)
                        if return_coordlist:
                            question_nums_found.append(question_num)
                            # make a tuple containing the question number, coordinates and the cropped image
                            list_cropped_images.append((question_num, [int(coord) for coord in xyxy], cropped_image))
                    else:
                        question_nums_found.append(-1)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(question_num, im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    else:

        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img or view_crop or return_coordlist or return_image:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if return_image:
                                annotator.box_label(xyxy, label, color=colors(c, True)) # hide this, because it gives a red line
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        if view_crop or return_coordlist:
                            cropped_image = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                                     BGR=True, save=False)
                            if view_crop:
                                cv2.imshow(str(conf), cropped_image)
                                cv2.waitKey(0)
                        if return_coordlist:
                            question_num = splitext(basename(path))[0]

                            question_nums_found.append(question_num)

                            # make a tuple containing the question number, coordinates and the cropped image
                            list_cropped_images.append((question_num, [int(coord) for coord in xyxy], cropped_image))

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    if return_coordlist:
        return list_cropped_images, question_nums_found

    if return_image:
        return im0


def show_image(img, window_height=500):
    """
    Display the cv2 image and close upon closing the window.
    :param img: cv2 image
    :param window_height: window height in pixels
    :return: empty
    """

    height = img.shape[0]
    width = img.shape[1]
    ratio = height / float(width)

    # set the dimensions
    dim = (window_height, int(window_height * ratio))

    img = cv2.resize(img, dim)
    cv2.imshow('Image', img)
    cv2.waitKey(0)



def find_cropped_images(image_list, question_list, weights=r"runs\train\exp45\weights\best_new.pt",
                        data=r"models\cito.yaml"):
    cropped_coord_list = run(weights=Path(weights),
                             imgsz=[640, 640],
                             max_det=1,
                             # conf_thres=0.3,
                             is_image_list=True,
                             question_nums=question_list,
                             iou_thres=0.1,  # threshold to avoid overlapping images
                             data=Path(data),
                             source=image_list,  # Path(file_list),
                             nosave=True,  # do/don't save images
                             save_crop=False,  # do/don't save the cropped images
                             view_img=False,
                             view_crop=False,
                             return_coordlist=True,
                             hide_labels=True,
                             classes=1)  # only do the handwritten

    cropped_coords = []

    for question in cropped_coord_list:
        cropped_coords.append(question[1])

    return cropped_coords



def main():

    # run(weights=Path(r"C:\Users\ajtis\Documents\Master\Thesis\pythonProject\yolov5\runs\train\exp45\weights\best_new.pt"),
    #     imgsz=[640,640],
    #     max_det=1,
    #     #conf_thres=0.3,
    #     iou_thres=0.1, # threshold to avoid overlapping images
    #     data=Path(r"C:\Users\ajtis\Documents\Master\Thesis\Deel Aico\cito.yaml"),
    #     source=Path(r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\Toetsen pdf's\toets1\Individual questions"),
    #     nosave=True, # do/don't save images
    #     save_crop=True, # do not save the cropped images
    #     classes=1) # only do the handwritten

    # dir_path = r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\Hogere resolutie toetsen\ToetsData"
    # folders = [join(dir_path, f) for f in listdir(dir_path) if isdir(join(dir_path, f))]
    #
    #
    # for index,folder in enumerate(folders):
    #     if index > 0:
    #         break
    #
    #
    #     file_list = join(folder, 'Individual questions')
    #
    #     path_list = [join(file_list, f) for f in listdir(file_list) if isfile(join(file_list, f))]
    #     question_list = [splitext(f)[0] for f in listdir(file_list) if isfile(join(file_list, f))]

        #print(question_list)

        #image_list = [cv2.imread(path) for path in path_list]

        # run(weights=Path(r"runs\train\exp45\weights\best_new.pt"),
        #     imgsz=[640,640],
        #     max_det=1,
        #     #conf_thres=0.3,
        #     iou_thres=0.1, # threshold to avoid overlapping images
        #     data=Path(r"C:\Users\ajtis\Documents\Master\Thesis\Deel Aico\cito.yaml"),
        #     source=Path(file_list),
        #     nosave=True, # do/don't save images
        #     save_crop=True, # do not save the cropped images
        #     classes=1) # only do the handwritten


        # cropped_coord_list = run(weights=Path(r"runs\train\exp45\weights\best_new.pt"),
        #                          imgsz=[640, 640],
        #                          max_det=1,
        #                          # conf_thres=0.3,
        #                          is_image_list=True,
        #                          question_nums=question_list,
        #                          iou_thres=0.1,  # threshold to avoid overlapping images
        #                          data=Path(r"models\cito.yaml"),
        #                          source=image_list,#Path(file_list),
        #                          nosave=True,  # do/don't save images
        #                          save_crop=True,  # do/don't save the cropped images
        #                          view_img=False,
        #                          view_crop=False,
        #                          return_coordlist=True,
        #                          classes=1)  # only do the handwritten

        #cropped_images = find_cropped_images(image_list, question_list)

    path = r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\Hogere_resolutie_toetsen\ToetsData\toets0\Improved questions 130\21.png"

    img = run(weights=Path(r"runs\train\exp45\weights\best_new.pt"),
                             imgsz=[640, 640],
                             max_det=1,
                             # conf_thres=0.3,
                             #is_image_list=True,
                             #question_nums=question_list,
                             iou_thres=0.1,  # threshold to avoid overlapping images
                             data=Path(r"models\cito.yaml"),
                             source=Path(path),#Path(file_list),
                             nosave=True,  # do/don't save images
                             save_crop=False,  # do/don't save the cropped images
                             view_img=False,
                             view_crop=False,
                             return_image=True,
                             #return_coordlist=True,
                             classes=1)  # only do the handwritten

    show_image(img)




if __name__ == "__main__":
    #opt = parse_opt()
    main() # main(opt)
