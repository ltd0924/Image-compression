import numpy as np
import struct
import os
import Haff
from collections import Counter

dc_huffman_table = {0: "00", 1: "010", 2: "011", 3: "100", 4: "101", 5: "110", 6: "1110", 7: "11110", 8: "111110",
                    9: "1111110",10: "11111110"}
run_huffman_table = {
(2, '1'):'000',
(1, 'Z'):'0010',
(4, '1'):'00110',
(7, '0'):'0011100',
(9, '0'):'00111010',
(10, 'Z'):'00111011000',
(42, '1'):'00111011001000',
(50, '1'):'001110110010010',
(54, '0'):'0011101100100110',
(52, '0'):'0011101100100111',
(26, '0'):'0011101100101',
(33, '1'):'0011101100110',
(43, '1'):'00111011001110',
(27, 'Z'):'001110110011110000',
(44, 'Z'):'0011101100111100010',
(29, 'Z'):'0011101100111100011',
(40, 'Z'):'00111011001111001000',
(39, 'Z'):'00111011001111001001',
(45, 'Z'):'001110110011110010100',
(35, 'Z'):'001110110011110010101',
(43, 'Z'):'0011101100111100101100',
(49, 'Z'):'0011101100111100101101',
(46, 'Z'):'001110110011110010111',
(37, 'Z'):'001110110011110011',
(48, '0'):'0011101100111101',
(39, '0'):'001110110011111',
(8, 'Z'):'0011101101',
(17, '0'):'00111011100',
(21, '1'):'00111011101',
(16, '1'):'0011101111',
(5, '0'):'001111',
(1, '1'):'01',
(1, '0'):'10',
(7, '1'):'1100000',
(5, 'Z'):'11000010',
(11, '0'):'110000110',
(16, 'Z'):'11000011100000',
(40, '1'):'11000011100001',
(41, '1'):'11000011100010',
(51, '1'):'110000111000110',
(47, '0'):'1100001110001110',
(51, '0'):'1100001110001111',
(26, '1'):'110000111001',
(20, '1'):'11000011101',
(37, '0'):'110000111100000',
(38, '0'):'110000111100001',
(35, '0'):'110000111100010',
(49, '1'):'110000111100011',
(57, '0'):'11000011110010000',
(62, '0'):'11000011110010001',
(44, '0'):'1100001111001001',
(18, 'Z'):'110000111100101',
(30, '0'):'11000011110011',
(32, '1'):'1100001111010',
(29, '0'):'11000011110110',
(38, '1'):'11000011110111',
(20, '0'):'110000111110',
(24, '0'):'1100001111110',
(45, '0'):'1100001111111000',
(58, '1'):'1100001111111001',
(46, '0'):'1100001111111010',
(54, '1'):'1100001111111011',
(48, '1'):'110000111111110',
(36, '0'):'110000111111111',
(5, '1'):'110001',
(9, '1'):'11001000',
(12, '1'):'110010010',
(6, 'Z'):'110010011',
(13, '0'):'1100101000',
(15, '1'):'1100101001',
(25, '1'):'110010101000',
(11, 'Z'):'110010101001',
(16, '0'):'11001010101',
(31, '1'):'1100101011000',
(47, '1'):'110010101100100',
(61, '0'):'11001010110010100',
(41, 'Z'):'1100101011001010100',
(38, 'Z'):'110010101100101010100',
(63, 'Z'):'1100101011001010101010',
(57, 'Z'):'11001010110010101010110',
(62, 'Z'):'11001010110010101010111',
(36, 'Z'):'11001010110010101011',
(30, 'Z'):'1100101011001010110',
(48, 'Z'):'11001010110010101110',
(56, 'Z'):'110010101100101011110',
(33, 'Z'):'110010101100101011111',
(59, '1'):'1100101011001011',
(39, '1'):'11001010110011',
(24, '1'):'110010101101',
(19, '1'):'11001010111',
(8, '0'):'11001011',
(6, '0'):'1100110',
(11, '1'):'110011100',
(10, '0'):'110011101',
(13, 'Z'):'1100111100000',
(56, '1'):'1100111100001000',
(60, '1'):'1100111100001001',
(34, '0'):'110011110000101',
(27, '0'):'11001111000011',
(46, '1'):'110011110001000',
(58, '0'):'11001111000100100',
(61, '1'):'11001111000100101',
(43, '0'):'1100111100010011',
(15, 'Z'):'11001111000101',
(23, '0'):'1100111100011',
(9, 'Z'):'11001111001',
(18, '1'):'11001111010',
(19, '0'):'110011110110',
(20, 'Z'):'1100111101110000',
(41, '0'):'1100111101110001',
(33, '0'):'110011110111001',
(37, '1'):'11001111011101',
(30, '1'):'1100111101111',
(14, '1'):'1100111110',
(23, '1'):'110011111100',
(42, '0'):'1100111111010000',
(19, 'Z'):'1100111111010001',
(45, '1'):'110011111101001',
(28, '0'):'11001111110101',
(40, '0'):'1100111111011000',
(55, '1'):'1100111111011001',
(52, 'Z'):'1100111111011010000',
(31, 'Z'):'1100111111011010001',
(64, 'Z'):'110011111101101001',
(49, '0'):'11001111110110101',
(57, '1'):'1100111111011011',
(35, '1'):'11001111110111',
(15, '0'):'11001111111',
(2, '0'):'1101',
(3, '0'):'11100',
(3, '1'):'11101',
(6, '1'):'1111000',
(3, 'Z'):'1111001',
(2, 'Z'):'111101',
(4, '0'):'111110',
(8, '1'):'11111100',
(7, 'Z'):'1111110100',
(12, '0'):'1111110101',
(18, '0'):'111111011000',
(64, '1'):'1111110110010',
(22, '0'):'1111110110011',
(22, '1'):'111111011010',
(53, '1'):'1111110110110000',
(53, '0'):'11111101101100010',
(56, '0'):'11111101101100011',
(44, '1'):'111111011011001',
(36, '1'):'11111101101101',
(29, '1'):'1111110110111',
(32, '0'):'111111011100000',
(17, 'Z'):'111111011100001',
(31, '0'):'111111011100010',
(42, 'Z'):'11111101110001100000',
(60, 'Z'):'111111011100011000010',
(32, 'Z'):'1111110111000110000110',
(55, 'Z'):'1111110111000110000111',
(34, 'Z'):'11111101110001100010',
(28, 'Z'):'11111101110001100011',
(26, 'Z'):'111111011100011001',
(50, '0'):'11111101110001101',
(63, '1'):'11111101110001110',
(23, 'Z'):'111111011100011110',
(59, '0'):'111111011100011111',
(28, '1'):'1111110111001',
(34, '1'):'11111101110100',
(14, 'Z'):'11111101110101',
(25, 'Z'):'111111011101100000',
(63, '0'):'111111011101100001',
(22, 'Z'):'11111101110110001',
(52, '1'):'1111110111011001',
(55, '0'):'11111101110110100',
(60, '0'):'111111011101101010',
(24, 'Z'):'111111011101101011',
(21, 'Z'):'11111101110110110',
(62, '1'):'11111101110110111',
(25, '0'):'11111101110111',
(17, '1'):'11111101111',
(4, 'Z'):'11111110',
(10, '1'):'111111110',
(13, '1'):'1111111110',
(12, 'Z'):'1111111111000',
(21, '0'):'1111111111001',
(27, '1'):'1111111111010',
(64, '0'):'1111111111011',
(14, '0'):'11111111111'
}
H0 = np.array([-1, 2, 6, 2, -1], dtype=float) / 8
H1 = np.array([-1, 2, -1], dtype=float) / 2

def raw_image(filename):
    img = np.fromfile(filename, dtype='uint8')
    img = img.reshape(512, 512)
    return img

def str2num(str):
    if len(str)<8:
        str = str+(8-len(str))*'0'
    return int(str, 2) - 128

def log(base, x):
    return np.log(x) / np.log(base)

def symfilter(X, F0, N):
    len_f = F0.shape[0]
    length = int((len_f - 1) / 2)
    return np.append(np.append(np.flipud(X[1: length + 1]), X), np.flipud(X[N - length - 1: N - 1]))

def DWTAnalysis(X, N):
    Y0 = symfilter(X, H0, N)
    Y1 = symfilter(X, H1, N)
    Y2 = np.zeros(N)
    Y3 = np.zeros(N)
    length = H0.shape[0]
    length1 = H1.shape[0]
    for i in range(N):
        Y2[i] = np.sum(Y0[i:i + length] * H0)
        Y3[i] = np.sum(Y1[i:i + length1] * H1)
    return np.append(Y2[::2], Y3[1::2])

def DWTAnalysis2D(X, R0, C0, R1, C1):
    N = R1 - R0
    for i in range(R0, R1):
        X[i, C0:C1] = DWTAnalysis(X[i, C0:C1], N)

    for i in range(C0, C1):
        X[R0:R1, i] = DWTAnalysis(X[R0:R1, i], N)
    return X

def quantizer(step_size, ori):
    return np.round(ori / step_size)

class EZT:
    def __init__(self, X, N, l):
        self.mask = np.zeros(np.shape(X), dtype=float)
        self.ori = X
        self.high_f_w = N
        self.E_symbol = ''
        self.E_nonzero = []
        self.H_num = []
        self.max_level = l
        self.raster_scan(0)
        #print(len(self.E_symbol))
        self.encode_E_symbol()
        #print(len(self.E_symbol))

    def calculate_symbol(self, x, y, ex, ey, l, D):
        for i in range(x, ex):
            for j in range(y, ey):
                if self.mask[i, j] == 1:
                    continue
                if self.ori[i, j] != 0:
                    self.E_symbol = self.E_symbol + '1'
                    self.mask[i, j] = 1
                    self.E_nonzero.append(self.ori[i, j])
                elif l != 0 and self.ZTR(i, j, 1, l, D):
                    self.E_symbol = self.E_symbol + 'Z'
                else:
                    self.E_symbol = self.E_symbol + '0'
                    self.mask[i, j] = 1

    def ZTR(self, x, y, level, cur, D):
        if level + cur == self.max_level:
            if np.all(self.ori[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1)] == 0) and level != 1:
                self.mask[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1)] = 1
                return True
            else:
                return False
        elif np.all(self.ori[x:x + 2 ** (level - 1), y:y + 2 ** (level - 1) ] == 0):
            if self.ZTR(D[0] * 2 ** level * self.high_f_w + 2*(x- D[0] * 2 ** (level - 1) * self.high_f_w),
                        D[1] * 2 ** level * self.high_f_w + 2*(y- D[1] * 2 ** (level - 1) * self.high_f_w),
                        level + 1, cur, D):
                self.mask[x:x + 2 ** (level - 1) , y:y + 2 ** (level - 1) ] = 1
                return True
            else:
                return False
        else:
            return False

    def encode_E_symbol(self):
        a = self.E_symbol[0]
        i = 0
        E_symbol_t = ''
        for x in self.E_symbol:
            if x == a and i < 64:
                i=i+1
            else:
                E_symbol_t= E_symbol_t+run_huffman_table[(i,a)]
                a = x
                i = 1
        E_symbol_t= E_symbol_t+run_huffman_table[(i,a)]
        self.E_symbol = E_symbol_t



    def predict_symbol(self, i, j, row, col):
        self.H_num.append(self.ori[i, j])
        while i < row and j < col:
            tt = self.ori[i, j]
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
            self.H_num.append(self.ori[i, j] - tt)
            if len(self.H_num)==col*row:
                break

    def raster_scan(self, l):
        if l == self.max_level:
            return
        if l == 0:
            self.predict_symbol(0, 0, self.high_f_w, self.high_f_w)
            self.raster_scan(l + 1)
        else:
            self.calculate_symbol(0, int(2 ** (l - 1) * self.high_f_w),
                                  int(2 ** (l - 1) * self.high_f_w), int(2 ** l * self.high_f_w), l, [0, 1])
            self.calculate_symbol(int(2 ** (l - 1) * self.high_f_w), 0,
                                  int(2 ** l * self.high_f_w), int(2 ** (l - 1) * self.high_f_w), l, [1, 0])

            self.calculate_symbol(int(2 ** (l - 1) * self.high_f_w), int(2 ** (l - 1) * self.high_f_w),
                                  int(2 ** l * self.high_f_w), int(2 ** l * self.high_f_w), l, [1, 1])
            self.raster_scan(l + 1)

    def non_zero_encode(self, str):
        # with open('img.bin', 'wb') as fp:
        e_str = ''
        for x in str:
            if x == 0:
                e_str = e_str + dc_huffman_table[0]
                continue
            c = int(log(2, abs(x)))+1
            e_str = e_str + dc_huffman_table[c]
            if x < 0:
                index = int(x + 2 ** c - 1)
            else:
                index = int(x)
            cc = format(index, 'b')
            cc = (c-len(cc))*'0'+cc
            e_str = e_str +cc
        return e_str

def ham_coding(E_symbol):
    ans = ''
    for x in E_symbol:
        if x == '1':
            ans = ans+'0'
        elif x =='0':
            ans = ans+'10'
        else:
            ans = ans+'11'
    return ans

def str2bin(stream, N, h_n, h_len_b, e_len_b, filename):
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', N))  # pic size
        f.write(struct.pack('i', h_n))  # pic high f size
        f.write(struct.pack('i', h_len_b))  # h_num len bit
        f.write(struct.pack('i', e_len_b))  # e_symbol len bit
        for i in range(0, len(stream), 8):
            f.write(struct.pack('b', str2num(stream[i:i + 8])))

def imageEncoder(org_file_name,q_size):
    c = raw_image(org_file_name)
    img = np.array(c, dtype='float64')
    img_ana = DWTAnalysis2D(img, 0, 0, 512, 512)
    img_ana[0:256, 0:256] = DWTAnalysis2D(img_ana[0:256, 0:256], 0, 0, 256, 256)
    img_ana[0:128, 0:128] = DWTAnalysis2D(img_ana[0:128, 0:128], 0, 0, 128, 128)
    img_ana[0:64, 0:64] = DWTAnalysis2D(img_ana[0:64, 0:64], 0, 0, 64, 64)
    img_ana[0:32, 0:32] = DWTAnalysis2D(img_ana[0:32, 0:32], 0, 0, 32, 32)
    q_img_ana = quantizer(q_size, img_ana)
    qq = EZT(q_img_ana, 16, 6)
    ans = qq.non_zero_encode(qq.E_nonzero)
    h_non = qq.non_zero_encode(qq.H_num)
    e_s = qq.E_symbol
    stream = h_non + e_s + ans
    str2bin(stream, 512, 16, len(h_non), len(e_s), "TEST.bin")
    size = os.path.getsize("TEST.bin")*8/512**2
    print(os.path.getsize("TEST.bin"))
    return size