import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import Haff
import math
import struct
import os

huffman_table = {"00": 0, "010":1, "011": 2, "100": 3, "101": 4, "110": 5, "1110": 6, "11110": 7, "111110": 8,
                 "1111110": 9,"11111110":10}
d_r_huffman_table={
'000':(2, '1'),'0010':(1, 'Z'),'00110':(4, '1'),'0011100':(7, '0'),'00111010':(9, '0'),'00111011000':(10, 'Z'),'00111011001000':(42, '1'),
    '001110110010010':(50, '1'),'0011101100100110':(54, '0'),'0011101100100111':(52, '0'),'0011101100101':(26, '0'),'0011101100110':(33, '1'),
    '00111011001110':(43, '1'),'001110110011110000':(27, 'Z'),'0011101100111100010':(44, 'Z'),'0011101100111100011':(29, 'Z'),'00111011001111001000':(40, 'Z'),
'00111011001111001001':(39, 'Z'),'001110110011110010100':(45, 'Z'),'001110110011110010101':(35, 'Z'),'0011101100111100101100':(43, 'Z'),'0011101100111100101101':(49, 'Z'),
'001110110011110010111':(46, 'Z'),'001110110011110011':(37, 'Z'),'0011101100111101':(48, '0'),'001110110011111':(39, '0'),'0011101101':(8, 'Z'),'00111011100':(17, '0'),
'00111011101':(21, '1'),'0011101111':(16, '1'),'001111':(5, '0'),'01':(1, '1'),'10':(1, '0'),'1100000':(7, '1'),'11000010':(5, 'Z'),'110000110':(11, '0'),
    '11000011100000':(16, 'Z'),'11000011100001':(40, '1'),'11000011100010':(41, '1'),'110000111000110':(51, '1'),'1100001110001110':(47, '0'),'1100001110001111':(51, '0'),
    '110000111001':(26, '1'),'11000011101':(20, '1'),'110000111100000':(37, '0'),'110000111100001':(38, '0'),'110000111100010':(35, '0'),'110000111100011':(49, '1'),
    '11000011110010000':(57, '0'),'11000011110010001':(62, '0'),'1100001111001001':(44, '0'),'110000111100101':(18, 'Z'),'11000011110011':(30, '0'),
    '1100001111010':(32, '1'),'11000011110110':(29, '0'),'11000011110111':(38, '1'),'110000111110':(20, '0'),'1100001111110':(24, '0'),'1100001111111000':(45, '0'),
    '1100001111111001':(58, '1'),'1100001111111010':(46, '0'),'1100001111111011':(54, '1'),'110000111111110':(48, '1'),'110000111111111':(36, '0'),'110001':(5, '1'),
    '11001000':(9, '1'),'110010010':(12, '1'),'110010011':(6, 'Z'),'1100101000':(13, '0'),'1100101001':(15, '1'),'110010101000':(25, '1'),'110010101001':(11, 'Z'),
    '11001010101':(16, '0'),'1100101011000':(31, '1'),'110010101100100':(47, '1'),'11001010110010100':(61, '0'),'1100101011001010100':(41, 'Z'),
    '110010101100101010100':(38, 'Z'),'1100101011001010101010':(63, 'Z'),'11001010110010101010110':(57, 'Z'),'11001010110010101010111':(62, 'Z'),
    '11001010110010101011':(36, 'Z'),'1100101011001010110':(30, 'Z'),'11001010110010101110':(48, 'Z'),'110010101100101011110':(56, 'Z'),'110010101100101011111':(33, 'Z'),
    '1100101011001011':(59, '1'),'11001010110011':(39, '1'),'110010101101':(24, '1'),'11001010111':(19, '1'),'11001011':(8, '0'),'1100110':(6, '0'),'110011100':(11, '1'),
    '110011101':(10, '0'),'1100111100000':(13, 'Z'),'1100111100001000':(56, '1'),'1100111100001001':(60, '1'),'110011110000101':(34, '0'),'11001111000011':(27, '0'),
    '110011110001000':(46, '1'),'11001111000100100':(58, '0'),'11001111000100101':(61, '1'),'1100111100010011':(43, '0'),'11001111000101':(15, 'Z'),
    '1100111100011':(23, '0'),'11001111001':(9, 'Z'),'11001111010':(18, '1'),'110011110110':(19, '0'),'1100111101110000':(20, 'Z'),'1100111101110001':(41, '0'),
    '110011110111001':(33, '0'),'11001111011101':(37, '1'),'1100111101111':(30, '1'),'1100111110':(14, '1'),'110011111100':(23, '1'),'1100111111010000':(42, '0'),
    '1100111111010001':(19, 'Z'),'110011111101001':(45, '1'),'11001111110101':(28, '0'),'1100111111011000':(40, '0'),'1100111111011001':(55, '1'),
    '1100111111011010000':(52, 'Z'),'1100111111011010001':(31, 'Z'),'110011111101101001':(64, 'Z'),'11001111110110101':(49, '0'),'1100111111011011':(57, '1'),
    '11001111110111':(35, '1'),'11001111111':(15, '0'),'1101':(2, '0'),'11100':(3, '0'),'11101':(3, '1'),'1111000':(6, '1'),'1111001':(3, 'Z'),'111101':(2, 'Z'),
    '111110':(4, '0'),'11111100':(8, '1'),'1111110100':(7, 'Z'),'1111110101':(12, '0'),'111111011000':(18, '0'),'1111110110010':(64, '1'),'1111110110011':(22, '0'),
    '111111011010':(22, '1'),'1111110110110000':(53, '1'),'11111101101100010':(53, '0'),'11111101101100011':(56, '0'),'111111011011001':(44, '1'),
    '11111101101101':(36, '1'),'1111110110111':(29, '1'),'111111011100000':(32, '0'),'111111011100001':(17, 'Z'),'111111011100010':(31, '0'),
    '11111101110001100000':(42, 'Z'),'111111011100011000010':(60, 'Z'),'1111110111000110000110':(32, 'Z'),'1111110111000110000111':(55, 'Z'),
    '11111101110001100010':(34, 'Z'),'11111101110001100011':(28, 'Z'),'111111011100011001':(26, 'Z'),'11111101110001101':(50, '0'),'11111101110001110':(63, '1'),
    '111111011100011110':(23, 'Z'),'111111011100011111':(59, '0'),'1111110111001':(28, '1'),'11111101110100':(34, '1'),'11111101110101':(14, 'Z'),
    '111111011101100000':(25, 'Z'),'111111011101100001':(63, '0'),'11111101110110001':(22, 'Z'),'1111110111011001':(52, '1'),'11111101110110100':(55, '0'),
    '111111011101101010':(60, '0'),'111111011101101011':(24, 'Z'),'11111101110110110':(21, 'Z'),'11111101110110111':(62, '1'),'11111101110111':(25, '0'),
    '11111101111':(17, '1'),'11111110':(4, 'Z'),'111111110':(10, '1'),'1111111110':(13, '1'),'1111111111000':(12, 'Z'),'1111111111001':(21, '0'),
    '1111111111010':(27, '1'),'1111111111011':(64, '0'),'11111111111':(14, '0')
}


G1 = np.array([-1, -2, 6, -2, -1], dtype=float) / 8
G0 = np.array([1, 2, 1], dtype=float) / 2

def raw_image(filename):
    img = np.fromfile(filename, dtype='uint8')
    img = img.reshape(512, 512)
    return img

def transnum(code):
    ans = int(code, 2)
    if code[0] == '0':
        ans = ans - 2 ** len(code) + 1
    return ans

def log(base, x):
    return np.log(x) / np.log(base)

def bin2str(filename):
    str_test = ''
    with open(filename, 'rb') as f:
        N = struct.unpack('i', f.read(4))  # pic size
        h_n = struct.unpack('i', f.read(4))  # pic high f size
        h_len_b = struct.unpack('i', f.read(4))  # e_symbol len bit
        e_len_b = struct.unpack('i', f.read(4))  # e_symbol len bit
        size = os.path.getsize(filename)
        size = size - 16
        for i in range(size):
            str_test = str_test + '{:0>8b}'.format(struct.unpack('b', f.read(1))[0] + 128)

        return str_test, N, h_n, h_len_b, e_len_b

def dehamffman_code(code_num, h_map,flag):
    ans_list = []
    i = 0
    g =1
    while i < len(code_num):
        if g == 0:
            break
        g = 0
        for j in range(9):
            if h_map.get(code_num[i:i + j]):
                a = h_map[code_num[i:i + j]]
                i = i + j
                if code_num[i:i + a]=='':
                    return ans_list
                ac = transnum(code_num[i:i + a])
                i = i + a
                ans_list.append(ac)
                g = 1
                break
            if flag==1 and code_num[i:i + j]=='00':
                ans_list.append(0)
                i = i + 2
                g = 1
                break
    return ans_list


class DEEZT:
    def __init__(self, N, h_w, e_s,non_zeros,h_s):
        self.mask = np.zeros((N, N), dtype=float)
        self.ori = np.zeros((N, N), dtype=float)
        self.high_f_w = h_w
        self.E_symbol= e_s
        self.E_nonzero = non_zeros
        self.H_num = h_s
        self.max_level = int(log(2, N / h_w))+1
        self.t = 0
        self.u = 0
        self.raster_scan(0)

    def de_predict_sym(self):
        self.ori[0,0] = self.H_num[0]
        i = 0
        j = 0
        row = self.high_f_w
        col = self.high_f_w
        k = 0
        while i < row and j < col:
            k = k+1
            tt = self.ori[i,j]
            if k == col*row:
                break
            if (i + j) % 2 == 0:
                if (i - 1) in range(row) and (j + 1) not in range(col):
                    i = i + 1
                elif (i - 1) not in range(row) and (j + 1) in range(col):
                    j = j + 1
                elif (i - 1) not in range(row) and (j + 1) not in range(col):
                    i = i + 1
                else:
                    i = i - 1
                    j = j + 1

            elif (i + j) % 2 == 1:
                if (i + 1) in range(row) and (j - 1) not in range(col):
                    i = i + 1
                elif (i + 1) not in range(row) and (j - 1) in range(col):
                    j = j + 1
                elif (i + 1) not in range(row) and (j - 1) not in range(col):
                    j = j + 1
                else:
                    i = i + 1
                    j = j - 1
            self.ori[i, j] = self.H_num[k]+ tt

    def calculate_symbol(self, x, y, ex, ey, l, D):
        x = int(x)
        y = int(y)
        ex = int(ex)
        ey = int(ey)
        for i in range(x, ex):
            for j in range(y, ey):
                if self.mask[i, j] == 1:
                    continue
                if self.E_symbol[self.t] == '0':
                    self.mask[i, j] = 1
                    self.ori[i, j] = 0
                elif self.E_symbol[self.t] == '1':
                    self.mask[i, j] = 1
                    try:
                        self.ori[i, j] = self.E_nonzero[self.u]
                    except:
                        print(i,j,self.u)
                    self.u = self.u + 1
                else:
                    self.EZTR(i, j, 1, l, D)
                self.t = self.t + 1

    def EZTR(self, x, y, level, cur, D):
        if level + cur == self.max_level:
            self.mask[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1) ] = 1
            self.ori[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1) ] = 0
        else:
            self.EZTR(D[0] * 2 ** level * self.high_f_w + 2*(x- D[0] * 2 ** (level - 1) * self.high_f_w),
                        D[1] * 2 ** level * self.high_f_w + 2*(y- D[1] * 2 ** (level - 1) * self.high_f_w),
                        level + 1, cur, D)
            self.ori[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1)] = 0
            self.mask[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1)] = 1

    def raster_scan(self, l):
        if l == self.max_level:
            return
        if l == 0:
            self.de_predict_sym()
            self.raster_scan(l+1)
        else:
            self.calculate_symbol(0, int(2 ** (l - 1) * self.high_f_w),
                                  int(2 ** (l - 1) * self.high_f_w), int(2 ** l * self.high_f_w), l, [0, 1])
            self.calculate_symbol(int(2 ** (l - 1) * self.high_f_w), 0,
                                  int(2 ** l * self.high_f_w), int(2 ** (l - 1) * self.high_f_w), l, [1, 0])

            self.calculate_symbol(int(2 ** (l - 1) * self.high_f_w), int(2 ** (l - 1) * self.high_f_w),
                                  int(2 ** l * self.high_f_w), int(2 ** l * self.high_f_w), l, [1, 1])
            self.raster_scan(l + 1)

def dequantizer(step_size, input):
    return np.round(input * step_size)

def symfilter(X, F0, N):
    len_f = F0.shape[0]
    length = int((len_f - 1) / 2)
    return np.append(np.append(np.flipud(X[1: length + 1]), X), np.flipud(X[N - length - 1: N - 1]))

def DWTSynthesis(X, N):
    length = G0.shape[0]
    length1 = G1.shape[0]
    Y0 = np.zeros(N)
    Y1 = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            Y0[i] = X[int(i / 2)]
        else:
            Y1[i] = X[int(i / 2 + N / 2 - 0.5)]
    Y2 = symfilter(Y0, G0, N)
    Y3 = symfilter(Y1, G1, N)
    for i in range(N):
        X[i] = np.sum(Y2[i:i + length] * G0) + np.sum(Y3[i:i + length1] * G1)
    return X

def DWTSynthesis2D(X, R0, C0, R1, C1):
    N = R1 - R0
    for i in range(C0, C1):
        X[R0:R1, i] = DWTSynthesis(X[R0:R1, i], N)
    for i in range(R0, R1):
        X[i, C0:C1] = DWTSynthesis(X[i, C0:C1], N)
    return X

def mse(img1,img2):
    return np.mean((img1 / 255. - img2 / 255.) ** 2)

def psnr(img1, img2):
    mse1=mse(img1,img2)
    print(mse1*255*255)
    if mse1 < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse1))

def decode_E_symbol(ac_haff,code_num):
    ans_list = ''
    i = 0
    g = 1
    while i < len(code_num):
        if g == 0:
            break
        g = 0
        for j in range(29):
            if ac_haff.get(code_num[i:i + j]):
                a = ac_haff[code_num[i:i + j]]
                ans_list=ans_list + a[0]*a[1]
                i = i+j
                g = 1
    return ans_list

def imageDecoder(encode_img, q_size, org_img,nn):
    stream1, N, h_n, h_len_b, e_len_b= bin2str(encode_img)
    h_e_s = stream1[0:h_len_b[0]]
    dee_s = stream1[h_len_b[0]:h_len_b[0] + e_len_b[0]]
    deans = stream1[h_len_b[0] + e_len_b[0]:-1]
    d_h_non = dehamffman_code(h_e_s, huffman_table, 1)
    dee_s1 = decode_E_symbol(d_r_huffman_table,dee_s)
    non_zeros = dehamffman_code(deans, huffman_table, 0)
    de_qq = DEEZT(N[0], h_n[0], dee_s1, non_zeros, d_h_non)
    de_img = dequantizer(q_size, de_qq.ori)
    de_img[0:32, 0:32] = DWTSynthesis2D(de_img[0:32, 0:32], 0, 0, 32, 32)
    de_img[0:64, 0:64] = DWTSynthesis2D(de_img[0:64, 0:64], 0, 0, 64, 64)
    de_img[0:128, 0:128] = DWTSynthesis2D(de_img[0:128, 0:128], 0, 0, 128, 128)
    de_img[0:256, 0:256] = DWTSynthesis2D(de_img[0:256, 0:256], 0, 0, 256, 256)
    de_img = DWTSynthesis2D(de_img, 0, 0, 512, 512)
    de_img = de_img.astype(np.uint8)
    c = raw_image(org_img)
    if not os.path.exists(nn):
        os.makedirs(nn)
    cv2.imwrite(nn+'/' + str(q_size) + '.png', de_img)

    return psnr(c, de_img)
