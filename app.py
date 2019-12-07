from flask import Flask, request, abort, render_template, render_template_string
from fastai.vision import *
from fastai.utils.mem import *
from fastai.callbacks.hooks import *
import cv2
import traceback
import mpld3

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, convert_mode='L')


class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom

def acc_image_seg(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

# for mac
# learn = load_learner(Path("/Users/kevinsattakun/PycharmProjects/image_segmentation_fastai"), 'unet_eye_seg_model.pkl')

#for pc
learn = load_learner(Path(r'C:\Users\kevin\PycharmProjects\image_segmentation_fastai'), 'unet_eye_seg_model.pkl')

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
            prediction[0].show(figsize=(5, 5))
            fig = plt.gcf()
            return render_template_string(mpld3.fig_to_html(fig))
        except Exception as e:
            traceback.print_exc()
            abort(500, str(e))

if __name__ == '__main__':
    app.run()