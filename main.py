# This is a sample Python script.
import csv
from tkinter import *
from tkinter import ttk
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import filedialog
from tkinter.filedialog import *

import cv2

from matplotlib import pyplot as plt, gridspec
import numpy as np
# import xlwt
# from xlsxwriter import Workbook

def massProcessing():
    global fileNames, img, temp_img
    fileNames = askopenfilenames(parent=window)
    fileNames = sorted(fileNames)
    output = format(text2.get("1.0",'end-1c'))
    sumRed_RAW = []
    sumViol_RAW = []
    sumGreen_RAW = []
    for i in range (len(fileNames)):
        if (fileNames[i].find("_FIR_") != -1):
            with open(output, 'w', newline='') as Kf:
                writer = csv.writer(Kf, delimiter=';')
                print(fileNames[i])
                img = cv2.imread(fileNames[i])
                gray = img[:, :, 1]  # Преобразование в градации серого
                threshold = np.mean(gray, axis=(0, 1))
                # Применение пороговой сегментации
                _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # Здесь 127 - пороговое значение
                print(threshold)
                # Находить контуры на бинаризованном изображении
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Создание маски для подсчета значений пикселей
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                # Рисуем контуры на маске
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Заполняем контуры маски
                # Получаем значения пикселей внутри маски
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                # Подсчет суммы значений всех пикселей внутри маски
                sum_pixel_values = np.sum(masked_img)
                print("Сумма значений всех пикселей внутри контуров после пороговой сегментации:", sum_pixel_values)
                # Для визуализации
                impair = np.concatenate((img[:, :, 1], masked_img[:, :, 1]), axis=1)
                plt.imshow(impair)
                plt.show()
                # Основной цикл оконного интерфейса
                # while True:
                #
                #
                #     key = cv2.waitKey(1) & 0xFF
                #
                #     # Прекращаем рисование при нажатии 'q'
                #     if key == ord('q'):
                #         break


                print("Сумма значений всех пикселей в контуре:", sum_pixel_values)
                points.clear()
                plt.close('all')
                sumRed = np.uint32(sum_pixel_values)
                sumRed_RAW.append(sumRed)
                print(sumRed)
        if (fileNames[i].find("_FIV_") != -1):
            with open(output, 'w', newline='') as Kf:
                writer = csv.writer(Kf, delimiter=';')
                print(fileNames[i])
                img = cv2.imread(fileNames[i])

                gray = img[:, :, 1]  # Преобразование в градации серого
                threshold = np.mean(gray, axis=(0, 1))
                # Применение пороговой сегментации
                _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # Здесь 127 - пороговое значение
                print(threshold)
                # Находить контуры на бинаризованном изображении
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Создание маски для подсчета значений пикселей
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                # Рисуем контуры на маске
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Заполняем контуры маски
                # Получаем значения пикселей внутри маски
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                # Подсчет суммы значений всех пикселей внутри маски
                sum_pixel_values = np.sum(masked_img)
                print("Сумма значений всех пикселей внутри контуров после пороговой сегментации:", sum_pixel_values)
                # Для визуализации
                impair = np.concatenate((img[:, :, 1], masked_img[:, :, 1]), axis=1)
                plt.imshow(impair)
                plt.show()
                # Основной цикл оконного интерфейса
                # while True:
                #
                #     key = cv2.waitKey(1) & 0xFF
                #
                #     # Прекращаем рисование при нажатии 'q'
                #     if key == ord('q'):
                #         break

                print("Сумма значений всех пикселей в контуре:", sum_pixel_values)
                points.clear()
                plt.close('all')
                sumViol = np.uint32(sum_pixel_values)
                sumViol_RAW.append(sumViol)
                print(sumViol)
        if (fileNames[i].find("_FIG_") != -1):
            with open(output, 'w', newline='') as Kf:
                writer = csv.writer(Kf, delimiter=';')
                print(fileNames[i])
                img = cv2.imread(fileNames[i])
                gray = img[:, :, 1]  # Преобразование в градации серого
                threshold = np.mean(gray, axis=(0, 1))
                # Применение пороговой сегментации
                _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # Здесь 127 - пороговое значение
                print(threshold)
                # Находить контуры на бинаризованном изображении
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Создание маски для подсчета значений пикселей
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                # Рисуем контуры на маске
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Заполняем контуры маски
                # Получаем значения пикселей внутри маски
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                # Подсчет суммы значений всех пикселей внутри маски
                sum_pixel_values = np.sum(masked_img)
                print("Сумма значений всех пикселей внутри контуров после пороговой сегментации:", sum_pixel_values)
                # Для визуализации
                impair = np.concatenate((img[:, :, 1], masked_img[:, :, 1]), axis=1)
                plt.imshow(impair)
                plt.show()
                # Основной цикл оконного интерфейса
                # while True:
                #
                #     key = cv2.waitKey(1) & 0xFF
                #
                #     # Прекращаем рисование при нажатии 'q'
                #     if key == ord('q'):
                #         break


                print("Сумма значений всех пикселей в контуре:", sum_pixel_values)
                points.clear()
                plt.close('all')
                sumGreen = np.uint32(sum_pixel_values)
                sumGreen_RAW.append(sumGreen)
                print(sumGreen)

        with open(output, 'w', newline='\n') as f:
            for j in range(min(len(sumRed_RAW), len(sumViol_RAW), len(sumGreen_RAW))):
                print(j)
                writer = csv.writer(f, delimiter=' ')
                writer.writerow([j, sumRed_RAW[j], sumViol_RAW[j], sumGreen_RAW[j]])
    plt.figure()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('FIV, FIR, unitless')
    plt.title('FIR, FIV vs time')
    x = np.linspace(1, len(sumRed_RAW),len(sumRed_RAW))
    plt.plot(x, sumRed_RAW, 'red', marker='o', label='FIR')
    x = np.linspace(1, len(sumViol_RAW), len(sumViol_RAW))
    plt.plot(x, sumViol_RAW, 'violet', marker='o', label='FIV')
    x = np.linspace(1, len(sumGreen_RAW), len(sumGreen_RAW))
    plt.plot(x, sumGreen_RAW, 'green', marker='o', label='FIG')
    plt.legend()
    plt.show()
       # Kf.close()
    text1.insert(INSERT, 'Готово')

# Глобальные переменные
drawing = False  # Флаг, определяющий, рисуем ли мы
ix, iy = -1, -1  # Начальная позиция мыши
points = []  # Список точек для рисования контура

# Функция для ручного рисования
def draw_circle(event, x, y, flags, param):
    global drawing, points, img, temp_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]  # Начинать с текущей позиции

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Создаем временное изображение для рисования
            img_copy = temp_img.copy()
            points.append((x, y))
            cv2.polylines(img_copy, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        cv2.polylines(img, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.imshow('Image', img)


def selectOutfile():
    Output = filedialog.askdirectory(parent=window)
    OutputF = Output + '/labs1.csv'
    text2.insert(INSERT, OutputF)

    # global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    # text1.delete(1.0, END)
    # text1.delete(1.0, END)
    # fileName = askopenfilenames(parent=window)
    # text1.insert(INSERT, fileName)
    # fileName = format(text1.get("1.0", 'end-1c'))
    # im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    # # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # print(type(im))
    # coef = int(text1.get(1.0, END))
    # im_multiplied = cv2.multiply(np.uint8(im), coef)
    # outputFilename = format(text1.get("1.0", 'end-1c')) + "_multiplied.png"
    #
    # cv2.imwrite(outputFilename, im_multiplied)
    # cv2.imshow(outputFilename, im_multiplied)
    # # plt.title('im multiplied')
    # # plt.colorbar()
    # # plt.show()
    # # fileName = fileName[:-3]
    # # plt.savefig(fileName + 'png')
    # # plt.show()
    # # outputFile = format(text3.get("1.0", 'end-1c'))
    return
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    window = Tk()
    window.geometry('1150x650')
    window.title("MassProcessing")
    text1 = Text(width=15, height=1)  # image
    text1.grid(column=1, row=1, sticky=W)
    text2 = Text(width=70, height=1)  # image
    text2.grid(column=1, row=0, sticky=W)
    btn1 = Button(window, text="Select Images", command=massProcessing)
    btn1.grid(column=0, row=1, sticky=W)
    btn2 = Button(window, text="Select SavePlace", command=selectOutfile)
    btn2.grid(column=0, row=0, sticky=W)
    window.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
