from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN


def draw_image_with_boxes(filename, result_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill = False, color = 'red')
        #ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color = 'red')
            ax.add_patch(dot)
    pyplot.show()

filename = 'woman1.jpg'

pixels = pyplot.imread(filename)

detector = MTCNN()

faces = detector.detect_faces(pixels)

draw_image_with_boxes(filename, faces)