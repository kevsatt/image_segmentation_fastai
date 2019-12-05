from flask import Flask, request, abort, render_template, send_file
from fastai.vision import *
from fastai.utils.mem import *
from fastai.callbacks.hooks import *
import cv2
import traceback

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, convert_mode='L')


class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom

def acc_image_seg(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

learn = load_learner(Path("/Users/kevinsattakun/PycharmProjects/image_segmentation_fastai"), 'unet_eye_seg_model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        """First, serving input form as a webpage"""
        return render_template("index.html")
    else:
        """Otherwise this is a POST to call the imported function"""
        print(request.files['file'])
        data = request.files['file']
        filename = data.filename
        data.save(filename)
        try:
            # test = Image(data)
            img = open_image(filename, convert_mode='L')
            prediction = learn.predict(img)
            # prediction[0].show(figsize=(5, 5))
            mask = prediction[0]
            mask.save('temp.png')
            mask_output = cv2.imread('temp.png')
            test = cv2.cvtColor(mask_output, cv2.COLOR_BGR2RGB)
            print(test)
            # test.save('test.png')
            return send_file('test.png', mimetype='image/gif')
        except Exception as e:
            traceback.print_exc()
            abort(500, str(e))

if __name__ == '__main__':
    app.run()