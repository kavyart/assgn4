import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io, color
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import cv2


def normalize(I):
    return (I - np.min(I)) / (np.max(I) - np.min(I))




############# INITIALS #############
def create_lightfield(I):
    lensletSize = 16
    (height, width, colors) = I.shape

    u = lensletSize                 # 16
    v = lensletSize                 # 16
    s = int(height / lensletSize)   # 400
    t = int(width / lensletSize)    # 700
    c = colors                      # 3

    lightfield = np.zeros((u,v,s,t,c))

    for i in range(u):
        for j in range(v):
            lightfield[i,j,:,:,:] = I[i::u,j::v,:]

    return lightfield


############# SUB-APERTURE VIEWS #############
def create_mosaic(lightfield):
    (u,v,s,t,c) = lightfield.shape

    mosaic = np.zeros((u*s,v*t,c))

    for i in range(u):
        for j in range(v):
            mosaic[i*s:(i+1)*s,j*t:(j+1)*t,:] = lightfield[i,j,:,:,:]

    return mosaic


############# REFOCUSING AND FOCAL-STACK SIMULATION #############
def focal_stack(lightfield, D):
    (u,v,s,t,c) = lightfield.shape
    stack_size = D.shape[0]

    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    U = np.arange(lensletSize) - maxUV
    V = np.arange(lensletSize) - maxUV

    x = np.arange(t)
    y = np.arange(s)

    focal_stack = np.zeros((s,t,c,stack_size))

    for i in range(u):
        for j in range(v):
            for color in range(c):
                image = lightfield[i,j,:,:,color] #+ (d * U[i]) + (d * V[j])
                interpolate_func = interpolate.interp2d(x, y, image)
                
                for k, d in enumerate(D):
                    # focal_stack[:,:,:,i] = focal_image(lightfield, d)
                    focal_stack[:,:,color,k] += interpolate_func(x + (d * V[j]), y - (d * U[i]))
                # focal_stack[:,:,color] += interpolate_func(x + (d * U[i]), y - (d * V[j]))

            # red = interpolate.interp2d(x, y, image[:,:,0])
            # blue = interpolate.interp2d(x, y, image[:,:,1])
            # green = interpolate.interp2d(x, y, image[:,:,2])

    focal_stack = focal_stack / (u*v)
    return focal_stack

# def focal_stack(lightfield):
#     D = np.arange(-0.5, 1.5, 0.1)
#     (u,v,s,t,c) = lightfield.shape
    
#     focal_stack = np.zeros((s,t,c,d))

#     for i, d in enumerate(D):
#         focal_stack[:,:,:,i] = focal_image(lightfield, d)



############# ALL-IN-FOCUS IMAGE AND DEPTH FROM FOCUS #############
def gamma_encoding(img):
    return np.where(img <= 0.0404482, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def sharpness_weight(I):
    (s,t,c,d) = I.shape
    sigma1 = 4
    sigma2 = 6

    I_xyz = np.zeros_like(I)
    w_sharpness = np.zeros((s,t,d))
    for i in range(d):
        I_xyz[:,:,:,i] = color.rgb2xyz(I[:,:,:,i])
        I_luminance = I_xyz[:,:,1,:]
        
        I_lowfreq = gaussian_filter(I_luminance, sigma1)
        I_highfreq = I_luminance - I_lowfreq
        w_sharpness = gaussian_filter(np.power(I_highfreq, 2), sigma2)

    return w_sharpness



def all_in_focus(lightfield):
    D = np.arange(-1.4, 0.6, 0.3)
    I = focal_stack(lightfield, D)
    (s,t,c,d) = I.shape
    w_sharpness = sharpness_weight(I)

    # mult = np.mult(w_sharpness, I.transpose())
    # print(mult.shape)
    I_allinfocus = np.zeros((s,t,c))
    # numerator = np.zeros((s,t,c))
    denominator = np.sum(w_sharpness, axis=-1)
    for color in range(c):
        numerator = np.zeros((s,t))
        for i in range(d):
        
        # numerator += (w_sharpness[:,:,i] * I[:,:,:,i].transpose((1,0,2)))
            numerator += (w_sharpness[:,:,i] * I[:,:,color,i])

        I_allinfocus[:,:,color] = numerator / denominator

    # numerator = np.sum(mult, axis=-1)
    # denominator = np.sum(w_sharpness, axis=-1)
    # I_allinfocus = np.divide(numerator, denominator)
    numerator = np.zeros((s,t))
    for i in range(d):
        numerator += (w_sharpness[:,:,i] * D[i])
    denominator = np.sum(w_sharpness, axis=-1)
    Depth = numerator / denominator

    return I_allinfocus, Depth

def depth_map(lightfield):
    D = np.arange(-1.4, 0.8, 0.3)
    I = focal_stack(lightfield, D)
    (s,t,c,d) = I.shape
    w_sharpness = sharpness_weight(I)

    numerator = np.zeros((s,t))
    for i in range(d):
        numerator += (w_sharpness[:,:,i] * D[i])
    denominator = np.sum(w_sharpness, axis=-1)
    Depth = numerator / denominator

    return Depth



############# FOCAL-APERTURE STACK AND CONFOCAL STEREO #############

def aperture_stack(lightfield):
    A = np.array([2, 4, 8, 16])
    D = np.arange(-1.3, 0.6, 0.3)
    astack_size = A.shape[0]
    fstack_size = D.shape[0]
    (u,v,s,t,c) = lightfield.shape

    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    U = np.arange(lensletSize) - maxUV
    V = np.arange(lensletSize) - maxUV

    x = np.arange(t)
    y = np.arange(s)

    focal_stack = np.zeros((s,t,c,astack_size,fstack_size))

    for k, a in enumerate(A):
        size_aper = 0
        for i in range(u):
            for j in range(v):
                if ((abs(U[i]) < (a/2)) and (abs(V[i]) < (a/2))):
                    size_aper += 1
                    for color in range(c):
                        image = lightfield[i,j,:,:,color]
                        interpolate_func = interpolate.interp2d(x, y, image)
                        
                        for l, d in enumerate(D):
                            focal_stack[:,:,color,k,l] += interpolate_func(x + (d * V[j]), y - (d * U[i]))

        focal_stack[:,:,:,k,:] = focal_stack[:,:,:,k,:] / size_aper
            
    return focal_stack


def create_collage(focal_aper_stack):
    (s,t,c,a,f) = focal_aper_stack.shape

    collage = np.zeros((a*s,f*t,c))

    for i in range(a):
        for j in range(f):
            collage[i*s:(i+1)*s,j*t:(j+1)*t,:] = focal_aper_stack[:,:,:,i,j]

    return collage



def AFI(focal_aper_stack, x, y):
    (s,t,c,a,f) = focal_aper_stack.shape
    
    I = focal_aper_stack[y,x,:,:,:]
    I = I.transpose((1,2,0))
    I = color.rgb2xyz(I)
    I_lum = I[:,:,1]

    return I_lum
    

def confocal_stereo(focal_aper_stack):
    (s,t,c,a,f) = focal_aper_stack.shape

    depth_map = np.zeros((s,t))

    for i in range(s):
        for j in range(t):
            I = AFI(focal_aper_stack, j, i)
            depth_map[i,j] = np.argmin(np.var(I, axis=0))

    depth_map = depth_map / (f - 1)

    return depth_map





############# REFOCUSING AN UNSTRUCTURED LIGHTFIELD #############


def refocus(frames, window_corners, g):
    (height, width) = g.shape
    ones = np.ones((height / width))
    gbar = np.full((height, width), np.mean(g))
    window_size = 160

    refocused_frames = []
    for k in range(len(frames)): 
        frame = frame[k]
        (x1,y1) = window_corners[k]

        I = (color.rgb2xyz(frame[y1:y1+window_size,x1:x1+window_size]))[:,:,1]
        I_height, I_width = I.shape
        Ibar = np.ones((I_height, I_width)) / (I_height * I_width) 

        gs_Ibar = signal.correlate2d(Ibar, g-gbar, mode="same")
        gs_I = signal.correlate2d(I, g-gbar, mode="same")
        numerator = gs_I + gs_Ibar 

        g_term = (g-gbar)*(g-gbar)
        I_squared = signal.correlate2d(I*I, ones, mode="same")
        I_Ibar = signal.correlate2d(I, Ibar, mode="same")
        I_term = I_squared - (2 * I_Ibar) + (Ibar * Ibar)
        denominator = g_term * I_term

        refocused_frame = I
        refocused_frame[y1:y1+window_size,x1:x1+window_size] = numerator/denominator
        refocused_frames.append(refocused_frame)

    # refocused_img = np.zeros_like(frames[0])
    # for k in range(len(refocused_frames)):
    #     frame = frames[k]
    #     refocused_frame = refocused_frames[k] 
    #     f_height, f_width = frame.shape
    #     y_shift, x_shift = np.unravel_index(np.argmax(refocused_frame), refocused_frame.shape)

    #     R = frame[:,:,0]
    #     G = frame[:,:,1]
    #     B = frame[:,:,2]
    #     x_input = np.arange(f_width)
    #     y_input = np.arange(f_height)
        
    #     Rf = interpolate.interp2d(x_input, y_input, R)
    #     Gf = interpolate.interp2d(x_input, y_input, G)
    #     Bf = interpolate.interp2d(x_input, y_input, B)

    #     x = x_input + x_shift
    #     y = y_input - y_shift

    #     Rs = Rf(x, y)
    #     Gs = Gf(x, y)
    #     Bs = Bf(x, y)

    #     shifted_frame = np.dstack((Rs, Gs, Bs))

    #     refocused_img += shifted_frame
    
    return refocused_img / len(refocused_frames)















def linearize(C):
    C_linear = np.where(C <= 0.0404482, C / 12.92, ((C + 0.055) / 1.055) ** 2.4)
    return C_linear








def main():
   
    print("Initializing variables...")
    N = 1
    
  
    # im = io.imread('assgn4/data/chessboard_lightfield.png')[::N, ::N] / 255
    # # width = 11200
    # # height = 6400
    # # dimensions: 6400 x 11200 x 3
    # print(im.shape)
    # fig = plt.figure()
    # plt.imshow(im)
    '''

    # lightfield = create_lightfield(im)
    # print(lightfield.shape)
    # np.save('lfield.npy', lightfield)
    lightfield = np.load('lfield.npy')
    print(lightfield.shape)

    # f_stack = focal_stack(lightfield)
    # print(f_stack.shape)
    # fig = plt.figure()
    # plt.imshow(f_stack[:,:,:,5])

    allinfocus, Depth = all_in_focus(lightfield)
    print(allinfocus.shape)
    # allinfocus = normalize(allinfocus)
    fig = plt.figure()
    plt.imshow(allinfocus)

    save = np.clip(allinfocus, 0, 1) * 255
    save_ubyte = save.astype(np.ubyte)
    io.imsave('allinfocus.png', save_ubyte)

    # Depth = depth_map(lightfield)
    # Depth = Depth / np.max(Depth)
    # np.set_printoptions(threshold=np.inf)
    # print(Depth)
    print(Depth.shape)
    print(Depth.max())
    print(Depth.min())
    # Depth = Depth.max - Depth
    Depth = 1 + Depth
    fig = plt.figure()
    plt.imshow(Depth, cmap='gray')

    save = np.clip(np.dstack([Depth, Depth, Depth]), 0, 1) * 255
    save_ubyte = save.astype(np.ubyte)
    io.imsave('depth.png', save_ubyte)

    # save = np.clip(Depth, 0, 1) * 255
    # save_ubyte = save.astype(np.ubyte)
    # io.imsave('depthmap.png', save_ubyte)



    '''
    # af_stack = aperture_stack(lightfield)
    # print(af_stack.shape)
    # np.save('afstack.npy', af_stack)
    af_stack = np.load('afstack.npy')
    print(af_stack.shape)
    # 0 0
    # 100 100
    pixel0 = AFI(af_stack, 30, 30)
    fig = plt.figure()
    plt.imshow(pixel0, cmap = 'gray')
    plt.savefig('pixel-30-30.png')
    pixel1 = AFI(af_stack, 250, 250)
    fig = plt.figure()
    plt.imshow(pixel1, cmap = 'gray')
    plt.savefig('pixel-250-250.png')
    pixel2 = AFI(af_stack, 100, 241)
    fig = plt.figure()
    plt.imshow(pixel2, cmap = 'gray')
    plt.savefig('pixel-100-241.png')

    confoc = confocal_stereo(af_stack)
    print(confoc.shape)
    fig = plt.figure()
    plt.imshow(confoc, cmap  = 'gray')
    plt.savefig('confoc_depth_map.png')

    # fig = plt.figure()
    # plt.imshow(af_stack[:,:,:,0,1])
    # fig = plt.figure()
    # plt.imshow(af_stack[:,:,:,1,1])
    # fig = plt.figure()
    # plt.imshow(np.abs(af_stack[:,:,:,3,1] - af_stack[:,:,:,2,1]))

    # collage = create_collage(af_stack)
    # print(collage.shape)
    # fig = plt.figure()
    # plt.imshow(collage)

    # save = np.clip(collage, 0, 1) * 255
    # save_ubyte = save.astype(np.ubyte)
    # io.imsave('collage.png', save_ubyte)


    # mosaic = create_mosaic(lightfield)
    # print(mosaic.shape)
    # fig = plt.figure()
    # plt.imshow(mosaic)

    # save = np.clip(mosaic, 0, 1) * 255
    # save_ubyte = save.astype(np.ubyte)
    # io.imsave('mosaic.png', save_ubyte)





    frames = []
    step = 50
    vidObj = cv2.VideoCapture("assgn4/data/video.mp4")
    frameExists = 1
    frameNum = 0
    while frameExists:
        frameExists, im = vidObj.read()
        if (frameNum % step == 0): frames.append(im)
        frameNum += 1



    plt.show()


    

if __name__ == "__main__":
    main()