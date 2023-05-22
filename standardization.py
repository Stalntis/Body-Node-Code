
import pandas as pd
#df = pd.DataFrame()
df = pd.read_csv("./数据/xyz坐标/x坐标.txt",header=None)    #读取第一行，不将第一行作为标题

x = (df-(-0.91))*255/(1.87)
#print(x)

x.to_csv("./数据/xyz标准化/x.csv", encoding='utf-8', header=None, index=False)#数据导出到tsetcsv.csv#


#y

df2 = pd.read_csv("./数据/xyz坐标/y坐标.txt",header=None)    #读取第一行，不将第一行作为标题
y = (df2-(-1.06))*255/(2.17)

y.to_csv("./数据/xyz标准化/y.csv", encoding='utf-8', header=None, index=False)#数据导出到tsetcsv.csv#

#z

df3 = pd.read_csv("./数据/xyz坐标/z坐标.txt",header=None)    #读取第一行，不将第一行作为标题
z = (df3-(-0.74))*255/(1.78)
#print(a)
z.to_csv("./数据/xyz标准化/z.csv", encoding='utf-8', header=None, index=False)#数据导出到tsetcsv.csv#

#合并xyz.csv为一个，输出为txt

f1 = pd.read_csv('./数据/xyz标准化/x.csv')
f2 = pd.read_csv('./数据/xyz标准化/y.csv')
f3 = pd.read_csv('./数据/xyz标准化/z.csv')
file = [f1, f2, f3]
all_data = pd.concat(file, axis=1)
all_data.to_csv("./数据/xyz标准化/合并rgb1" + ".csv", index=0, sep=',')
#取整

df = pd.read_csv('./数据/xyz标准化/合并rgb1.csv'.format(),header=None)
df.astype(int).to_csv("./rgb.txt", encoding='utf-8', header=None, index=False)#数据导出到tsetcsv.csv#
df.astype(int).to_csv("./rgb.csv", encoding='utf-8', header=None, index=False)#数据导出到tsetcsv.csv#

#替换txt文件中逗号为空格
#替换txt文件中逗号为空格

lines = open('rgb.txt').readlines()
fp = open('rgb.txt','w')

for s in lines:
    fp.write( s.replace(',',' '))   #，替换为空格
fp.close()

#图片
count = len(open("rgb.txt",'r').readlines())
a=int(float(count/33))

print(a)

from PIL import Image
import math

x =  a	 #width    #x横坐标  通过对txt里的行数进行整数分解
y =	  33		#height    #y纵坐标  x * y = 行数     #按数据顺序第一列从上到下
im = Image.new("RGB", (x, y))   #创建图片
file = open('rgb.txt')    #打开rbg值的文件
#通过每个rgb点生成图片
for i in range(0, x):
	for j in range(0, y):
		line = file.readline()  #获取一行的rgb值
		rgb = line.split(" ")  #分离rgb，文本中逗号后面有空格
		im.putpixel((i,j), (int(rgb[0]), int(rgb[1]), int(rgb[2])))  # 将rgb转化为像素



im.save("Image original image address")   #im.save('flag.jpg')保存为jpg图片
#im.show()


img_pil = Image.open("Image original image address")
img_pil = img_pil.resize((32, 32))
#print(type(img_pil))       # <class 'PIL.Image.Image'>
#img_pil.show()
img_pil.save("Image address after resize")







