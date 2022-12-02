import Haff
from client.Encoder import imageEncoder
from server.Decoder import imageDecoder

q_size = [1,3,5,7,9,13,17,20,22,24,26,28,30,32,34,36,38,40]
source_data = ['./client/source_data/image1.512','./client/source_data/image2.512','./client/source_data/image3.512',
               './client/source_data/image4.512','./client/source_data/image5.512']
R=[]
D = []
R.append(imageEncoder(source_data[0],1))
D.append(imageDecoder('./TEST.bin',1,source_data[0],'./image1'))