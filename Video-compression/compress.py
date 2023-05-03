import math
import cv2
import sys
import os

# Exception handling
#argument check
if len(sys.argv) != 3:
	raise Exception('Invariant Number of Arguments passed.Please pass one video path and one compress percentage')

#file type check
t = str(sys.argv[1]).split(".")
if t[1] not in ["mp4","wav"]:
	raise Exception('Raise Exception incompatible file type only mp4 or wav required')

#range of compression check
if int(sys.argv[2]) not in range(1,100):
	raise Exception('Compress percent should be in range 1-99')

# file existence check
if not os.path.exists(sys.argv[1]):
	raise Exception('the video file does not exists or the path is incorrect')


# resizing all frames
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim)

# #Importing Decoder 
# from Decoder import CalcuPSNR
# from Decoder import load_graph
# from Decoder import decoder

# #Importing Encoder
# from Decoder import encoder

# getting video and then processing it and saving in filename_ouput.mp4
cap = cv2.VideoCapture(str(sys.argv[1]))
width  = (cap.get(3) * int(sys.argv[2]))/ 100
height = (cap.get(4) * int(sys.argv[2]))/ 100

def encoder(loadmodel, input_path, refer_path, outputfolder):
    graph = load_graph(loadmodel)
    prefix = 'import/build_towers/tower_0/train_net_inference_one_pass/train_net/'

    Res = graph.get_tensor_by_name(prefix + 'Residual_Feature:0')
    inputImage = graph.get_tensor_by_name('import/input_image:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')
    Res_prior = graph.get_tensor_by_name(prefix + 'Residual_Prior_Feature:0')
    motion = graph.get_tensor_by_name(prefix + 'Motion_Feature:0')
    bpp = graph.get_tensor_by_name(prefix + 'rate/Estimated_Bpp:0')
    psnr = graph.get_tensor_by_name(prefix + 'distortion/PSNR:0')
    
    reconframe = graph.get_tensor_by_name(prefix + 'ReconFrame:0')

def decoder(loadmodel, refer_path, outputfolder):
    graph = load_graph(loadmodel)

    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/train_net_inference_one_pass/train_net/ReconFrame:0')
    res_input = graph.get_tensor_by_name('import/quant_feature:0')
    res_prior_input = graph.get_tensor_by_name('import/quant_z:0')
    motion_input = graph.get_tensor_by_name('import/quant_mv:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')


out_video = cv2.VideoWriter(t[0]+'_compressed.mp4',0x7634706d, 20.0, (int(width), int(height)),True)
while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frameX = rescale_frame(frame,int(sys.argv[2]))
            out_video.write(frameX) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()