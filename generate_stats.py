import matplotlib.pyplot as plt
import pickle

base_dir = "./stats/"
file_names = [
"BUZZ_ann_stats.pickle",
"BUZZ_cn_stats.pickle",
"image_ann_stats.pickle",
"image_cn_stats.pickle"
]

x_plots = ["8", "16", "32"]
buzz_1_ann_plots = []
buzz_2_ann_plots = []
buzz_3_ann_plots = []
buzz_1_cn_plots = []
buzz_2_cn_plots = []
buzz_3_cn_plots = []

image_1_ann_plots = []
image_2_ann_plots = []
image_4_ann_plots = []

image_1_cn_plots = []
image_2_cn_plots = []
image_4_cn_plots = []

i = 0
for file_name in file_names:
    with open(base_dir+file_name, "rb") as file:
        model_stats = pickle.load(file)
        for key, value in model_stats.items():
            print(key,value)
            if i < 9:
                if "BUZZ1" in key:
                    buzz_1_ann_plots.append(value)
                if "BUZZ2" in key:
                    buzz_2_ann_plots.append(value)
                if "BUZZ3" in key:
                    buzz_3_ann_plots.append(value)
            elif i < 18:
                if "BUZZ1" in key:
                    buzz_1_cn_plots.append(value)
                if "BUZZ2" in key:
                    buzz_2_cn_plots.append(value)
                if "BUZZ3" in key:
                    buzz_3_cn_plots.append(value)
            elif i < 27:
                if "bee1" in key:
                    image_1_ann_plots.append(value)
                if "bee2" in key:
                    image_2_ann_plots.append(value)
                if "bee4" in key:
                    image_4_ann_plots.append(value)
            elif i < 36:
                if "bee1" in key:
                    image_1_cn_plots.append(value)
                if "bee2" in key:
                    image_2_cn_plots.append(value)
                if "bee4" in key:
                    image_4_cn_plots.append(value)
            i += 1
# create plots after that messy file extraction lol
print(buzz_1_ann_plots)

plt.title("BUZZ 1 Dataset ANN")
plt.bar(x_plots, buzz_1_ann_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("BUZZ 2 Dataset ANN")

plt.bar(x_plots, buzz_2_ann_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("BUZZ 3 Dataset ANN")

plt.bar(x_plots, buzz_3_ann_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("BUZZ 1 Dataset CNN")

plt.bar(x_plots, buzz_1_cn_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("BUZZ 2 Dataset CNN")

plt.bar(x_plots, buzz_2_cn_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("BUZZ 3 Dataset CNN")

plt.bar(x_plots, buzz_3_cn_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

# images

plt.title("bee 1 Dataset ANN")
plt.bar(x_plots, image_1_ann_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("bee 2 Dataset ANN")

plt.bar(x_plots, image_2_ann_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("bee 4 Dataset ANN")

plt.bar(x_plots, image_4_ann_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("bee 1 Dataset CNN")

plt.bar(x_plots, image_1_cn_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("bee 2 Dataset CNN")

plt.bar(x_plots, image_2_cn_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()

plt.title("bee 4 Dataset CNN")

plt.bar(x_plots, image_4_cn_plots)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()
