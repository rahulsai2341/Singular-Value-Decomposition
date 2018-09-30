import imageio
import numpy as np
import matplotlib.pyplot as plt
#Read image
rgb_image = imageio.imread('C:***.png');
rgb_image = rgb_image.astype(np.float64);
#Store SVD result of each channel
ur,sr,hr = np.linalg.svd(rgb_image[:,:,0])
ug,sg,hg = np.linalg.svd(rgb_image[:,:,1])
ub,sb,hb = np.linalg.svd(rgb_image[:,:,2])

#plot loglog graph of red channel
'''plt.loglog(sr)
plt.ylabel('Singular values of R')
plt.show()'''

#Calculate the frobenius norm of error matrix of all the three channels
normR = []
normG = []
normB = []
for i in range(0,500):
    reconstimgR = np.matmul(ur[:, :i],(np.matmul(np.diag(sr[:i]),(hr[:i, :]))))
    reconstimgG = np.matmul(ug[:, :i],(np.matmul(np.diag(sg[:i]),(hg[:i, :]))))
    reconstimgB = np.matmul(ub[:, :i],(np.matmul(np.diag(sb[:i]),(hb[:i, :]))))
    diffmatR = (rgb_image[:,:,0]) - reconstimgR
    diffmatG = (rgb_image[:,:,1]) - reconstimgG
    diffmatB = (rgb_image[:,:,2]) - reconstimgB
    normR.append(np.linalg.norm(diffmatR,'fro'))
    normG.append(np.linalg.norm(diffmatG, 'fro'))
    normB.append(np.linalg.norm(diffmatB, 'fro'))

#Plot the frobenius norms
x = np.linspace(1,200,200)
ax1 = plt.subplot(311)
plt.plot(x,normR)
ax2 = plt.subplot(312,sharex=ax1)
plt.plot(x,normG)
ax3 = plt.subplot(313,sharex=ax1)
plt.plot(x,normB)
plt.show()

#Reconstruct the image
reconstimg = np.zeros((2000,2000,3));
reconstimg[:,:,0] = reconstimgR
reconstimg[:,:,1] = reconstimgG
reconstimg[:,:,2] = reconstimgB

#Write image to disk
imageio.imwrite("***.jpg",reconstimg)