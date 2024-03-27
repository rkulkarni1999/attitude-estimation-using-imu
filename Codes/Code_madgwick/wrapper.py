########################
## Importing Libraries
########################

import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.animation as animation
from rotplot import rotplot

###############
## FUNCTIONS ##
###############

##########################################
## Angular Velocity Vector to Quaternion
##########################################

def ang_vel_to_quat(ang_vel_vector):

    # Compute Alpha
    alpha = np.linalg.norm(ang_vel_vector)
    print(f"Alpha : {alpha}")
    # Compute e_bar
    e_bar = ang_vel_vector / alpha


    # Calculating the Quaternion Components.
    w = np.cos(alpha/2)
    x = e_bar[0] * np.sin(alpha/2)
    y = e_bar[1] * np.sin(alpha/2)
    z = e_bar[2] * np.sin(alpha/2)

    # Returning the Quaternion as a numpy array
    return np.array([w, x, y, z])

###############################################
## Angular Velocity Vector to Quaternion Delta
###############################################

def ang_vel_to_quat_delta(ang_vel_vector, delta_t):

    # Compute alpha delta
    alpha_delta = np.linalg.norm(ang_vel_vector) * delta_t

    # Compute e_bar
    e_bar_delta = ang_vel_vector / np.linalg.norm(ang_vel_vector)

    # Calculating the Quaternion Components.
    w = np.cos(alpha_delta/2)
    x = e_bar_delta[0] * np.sin(alpha_delta/2)
    y = e_bar_delta[1] * np.sin(alpha_delta/2)
    z = e_bar_delta[2] * np.sin(alpha_delta/2)

    # Returning the Quaternion as a numpy array
    return np.array([w, x, y, z])

##########################################
## Quaternion Multiplication
##########################################

def quaternion_multiply(q1, q2):

    # Extracting coeffs from Quaternion
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Performing Quaternion Multiplication.
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Returning a Numpy array
    return np.array([w, x, y, z])

##########################################
## Quaternion to RPY Converter
##########################################

def quaternion_to_rpy(quaternion):

    ## Extracting coeffs
    q0,q1,q2,q3 = quaternion
    ## Roll
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    ## Pitch
    sin_pitch = 2 * (q0 * q2 - q3 * q1)
    if np.abs(sin_pitch) >= 1:
        pitch = np.sign(sin_pitch) * np.pi / 2
    else:
        pitch = np.arcsin(sin_pitch)
    ## Yaw
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

    return np.array([roll, pitch, yaw])

#############
# IMU Data
#############

########################
## Train Set
########################

imu_raw_data1 = io.loadmat(f"Data/Train/IMU/imuRaw6.mat")

# Extracting values and timesteps from IMU Data.
imu_vals1 = imu_raw_data1['vals']
imu_ts1 = imu_raw_data1['ts'].flatten().astype('float64')

######################
## VICON DATA
######################

vicon_true_data1 = io.loadmat(f"Data/Train/Vicon/viconRot6.mat")

vicon_rots1 = vicon_true_data1['rots']
vicon_ts1 = vicon_true_data1['ts']

#############################
## ACCELERATION PARAMETERS
#############################

# Loading Accelerometer Parameters.

imu_acc_params = io.loadmat("Data/IMUParams.mat")
imu_params = imu_acc_params["IMUParams"]

# Extracting Scales and Biases
imu_scales = imu_params[0,:]
imu_biases = imu_params[1,:]

##############################################################
## FUNCTION FOR CONVERTING RAW MATRICES TO APPRORIATE UNITS
##############################################################

def convert_raw_to_appropriate_units(imu_vals, imu_scales, imu_biases):

    # Extracting Scales and Biases
    imu_scales = imu_params[0,:]
    imu_biases = imu_params[1,:]

    # Extracting Biases and Scale Factors for Accelerometer Calculations.
    sx, sy, sz = imu_scales
    ba_x, ba_y, ba_z = imu_biases

    # Computing Bg values
    bg_x = imu_vals[4,:][:200].mean()
    bg_y = imu_vals[5,:][:200].mean()
    bg_z = imu_vals[3,:][:200].mean()

    # Created a copy to store converted values.
    imu_converted_vals = np.around(imu_vals.copy().astype('float64'), decimals=5)

    # Converting the Acceleration Matrix to m/s^2
    imu_converted_vals[0,:] = imu_converted_vals[0,:]*sx + ba_x
    imu_converted_vals[1,:] = imu_converted_vals[1,:]*sy + ba_y
    imu_converted_vals[2,:] = imu_converted_vals[2,:]*sz + ba_z

    # Converting The Angular Matrix to rad/sec
    imu_converted_vals[3,:] = (3300 / 1023) * (np.pi / 180) * 0.3 * (imu_vals[4,:] - bg_x)
    imu_converted_vals[4,:] = (3300 / 1023) * (np.pi / 180) * 0.3 * (imu_vals[5,:] - bg_y)
    imu_converted_vals[5,:] = (3300 / 1023) * (np.pi / 180) * 0.3 * (imu_vals[3,:] - bg_z)

    return imu_converted_vals

#################################################
## ATTITUDE ESTIMATION USING PURE ACCELEROMETER
#################################################

def attitude_estimation_accelerometer(imu_converted_vals, imu_ts):

    linear_accelerations = imu_converted_vals[:3,:]

    euler_from_accelerations = np.empty(imu_converted_vals[:3,:].shape)

    euler_from_accelerations[0,:] = np.arctan2( linear_accelerations[1,:] , np.sqrt((linear_accelerations[0,:]**2) + (linear_accelerations[2,:]**2)) )
    euler_from_accelerations[1,:] = np.arctan2( -linear_accelerations[0,:] , np.sqrt((linear_accelerations[1,:]**2) + (linear_accelerations[2,:]**2)) )
    euler_from_accelerations[2,:] = np.arctan2( np.sqrt((linear_accelerations[0,:]**2)+(linear_accelerations[1,:]**2)) , linear_accelerations[2,:] )

    euler_from_accelerations = euler_from_accelerations.T

    roll_acc = euler_from_accelerations[:,0]
    pitch_acc = euler_from_accelerations[:,1]
    yaw_acc = euler_from_accelerations[:,2] * 0

    return euler_from_accelerations, roll_acc, pitch_acc, yaw_acc

#################################################
## ATTITUDE ESTIMATION USING PURE GYROSCOPE
#################################################

def attitude_estimation_gyroscope(imu_converted_vals, imu_ts):

    # Extracting Angular Velocities and timestamps from Raw Data.
    angular_velocities = imu_converted_vals[3:,:]
    timestamps = np.reshape(imu_ts,(1,imu_ts.shape[0]))

    # Assuming Initial Quaternion at rest
    quaternion_k = np.array([1,0,0,0])
    time_k = timestamps[:,0]

    # Initializing All Quaternions
    all_quaternions = np.empty((angular_velocities.shape[1], 4))
    all_quaternions[0,:] = np.reshape(quaternion_k, (1,4))

    # Initializing All Euler Angles
    euler_from_gyro = np.empty((angular_velocities.shape[1], 3))

    for ii in range(angular_velocities.T.shape[0]):

        time_k_plus_1 = timestamps[:,ii]
        dt = time_k_plus_1 - time_k
        time_k = time_k_plus_1

        delta_quaternion = ang_vel_to_quat_delta(angular_velocities[:,ii], dt)
        quaternion_k_plus_1 = quaternion_multiply(quaternion_k, delta_quaternion)

        quaternion_k = quaternion_k_plus_1
        all_quaternions[ii,:] = quaternion_k_plus_1.T
        rpy = quaternion_to_rpy(quaternion_k_plus_1)
        euler_from_gyro[ii,:] = rpy.T

    # Extracting Roll, Pitch and Yaw
    roll_gyro = euler_from_gyro[:,0]
    pitch_gyro = euler_from_gyro[:,1]
    yaw_gyro = euler_from_gyro[:,2]

    return euler_from_gyro, roll_gyro, pitch_gyro, yaw_gyro

####################################################################
## FUNCTIONS FOR CONVERTING TRUE DATA FROM VICON TO EULER ANGLES
#####################################################################

def vicon_to_rpy(vicon_rots):

    euler_from_vicon = np.empty((3, vicon_rots.shape[2]))

    for ii in range(vicon_rots.shape[2]):

        rotation_matrix = vicon_rots[:,:,ii]

        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        euler_from_vicon[2, ii] = roll
        euler_from_vicon[1, ii] = pitch
        euler_from_vicon[0, ii] = yaw

    euler_from_vicon = euler_from_vicon.T

    roll_vicon = euler_from_vicon[:, 0]
    pitch_vicon = euler_from_vicon[:, 1]
    yaw_vicon = euler_from_vicon[:, 2]

    return roll_vicon, pitch_vicon, yaw_vicon

####################################################
## ATTITUDE ESTIMATION USING COMPLEMENTARY FILTER
####################################################

def attitude_estimation_complementary_filter(euler_from_accelerations, euler_from_gyro ,alpha=0.75, beta=0.75, gamma=0.1):

    mixing_parameters = np.array([alpha,beta,gamma])

    mixing_param_mat = np.diag(mixing_parameters)
    mixing_param_comp_mat = np.identity(3) - mixing_param_mat

    euler_from_comp = (mixing_param_mat @ euler_from_accelerations.T) + (mixing_param_comp_mat @ euler_from_gyro.T)


    # Extracting RPY from Matrix
    roll_comp = euler_from_comp[0,:]
    pitch_comp = euler_from_comp[1,:]
    yaw_comp = euler_from_comp[2,:]

    return euler_from_comp,roll_comp, pitch_comp, yaw_comp


####################################################
## ATTITUDE ESTIMATION USING MADGWICK FILTER
####################################################

def madgwick_filter_function(imu_converted_vals, imu_ts, beta):

  # Extracting Gyroscope and Accelerometer measurements.
  angular_velocities = imu_converted_vals[3:,:]          # Angular Velocities
  accelerations = imu_converted_vals[3:,:]               # Linear Accelerations
  timestamps = np.reshape(imu_ts,(1,imu_ts.shape[0]))    # Timestamps

  # Assuming Initial Quaternion at rest.
  quaternion_est = np.array([1.0,0.0,0.0,0.0])

  # Current time
  time_k = timestamps[:,0]

  # Initialize All Quaternions
  all_quaternions_madgwick = np.empty((imu_converted_vals.shape[1], 4))
  all_quaternions_madgwick[0,:] = np.reshape(quaternion_est, (1,4))

  # Initializing All Euler Angles
  euler_from_madgwick = np.empty((imu_converted_vals.shape[1], 3))

  for ii in range(1,angular_velocities.T.shape[0]):

    time_k_plus_1 = timestamps[:,ii]
    dt = time_k_plus_1 - time_k
    time_k = time_k_plus_1

    # Extracting Measurement
    gyro_measurement = angular_velocities[:,ii]
    acc_measurement = accelerations[:,ii]

    # Normalizing Acceleration Measurements [Not yet determined if necessary or not]
    norm_acc_measurement = acc_measurement / np.linalg.norm(acc_measurement)

    # Step 2(a): Compute orientation increment from acc measurements
    f_gradient = np.array([
        [2 * (quaternion_est[1] * quaternion_est[3] - quaternion_est[2] * quaternion_est[0]) - norm_acc_measurement[0]],
        [2 * (quaternion_est[0] * quaternion_est[1] + quaternion_est[2] * quaternion_est[3]) - norm_acc_measurement[1]],
        [2 * (0.5 - quaternion_est[1]**2 - quaternion_est[2]**2) - norm_acc_measurement[2]]
    ])

    jacobian = np.array([
        [-2 * quaternion_est[2], 2 * quaternion_est[3], -2 * quaternion_est[0], 2 * quaternion_est[1]],
        [2 * quaternion_est[1], 2 * quaternion_est[0], 2 * quaternion_est[3], 2 * quaternion_est[2]],
        [0, -4 * quaternion_est[1], -4 * quaternion_est[2], 0]
    ])

    jacobian_transpose = jacobian.T

    # delta_q_acc = -beta * (jacobian_transpose @ f_gradient) / (np.linalg.norm(f_gradient))
    delta_q_acc = -beta * (jacobian_transpose @ f_gradient) / (np.linalg.norm(jacobian_transpose @ f_gradient))

    # Calculating Orientation Increment from Gyroscope.
    gyro_quaternion = np.concatenate(([0], gyro_measurement))
    delta_q_gyro = 0.5 * (quaternion_multiply(quaternion_est, gyro_quaternion))

    # delta_q_gyro = delta_q_gyro.reshape(-1,1)

    # Fusing Measurements to obtain estimated attitude.
    delta_q_est = (delta_q_gyro + delta_q_acc.T)*dt

    quaternion_est_k_plus1 = quaternion_est + delta_q_est[0,:]
    quaternion_est_k_plus1 /= np.linalg.norm(quaternion_est_k_plus1)  # Normalize quaternion

    # Updating the quaternion for the next iteration.
    quaternion_est = quaternion_est_k_plus1 # For the next Iteration.

    # Appending Quaternion to the List.
    all_quaternions_madgwick[ii,:] = quaternion_est_k_plus1

    # Converting the quaternion to Euler Angles.
    rpy_madgwick = quaternion_to_rpy(quaternion_est_k_plus1)

    # Appending the Euler Angle to the Euler Angles List.
    euler_from_madgwick[ii,:] = rpy_madgwick

  # Converting Quaternions to Euler Angles.
  roll_madgwick = euler_from_madgwick[:,0]
  pitch_madgwick = euler_from_madgwick[:,1]
  yaw_madgwick = euler_from_madgwick[:,2]

  return euler_from_madgwick, roll_madgwick, pitch_madgwick, yaw_madgwick

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Define rotation matrices for each axis
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    # Combine the rotation matrices in the specified order
    rotation_matrix = np.dot(np.dot(R_yaw, R_pitch), R_roll)

    return rotation_matrix

########################
## DRIVER CODE 
########################

# Computing RPY
imu_converted_vals1 = convert_raw_to_appropriate_units(imu_vals1, imu_scales, imu_biases)
euler_from_accelerations1, roll_acc1, pitch_acc1, yaw_acc1 = attitude_estimation_accelerometer(imu_converted_vals1, imu_ts1)
euler_from_gyro1, roll_gyro1, pitch_gyro1, yaw_gyro1 = attitude_estimation_gyroscope(imu_converted_vals1, imu_ts1)
euler_from_comp1,roll_comp1, pitch_comp1, yaw_comp1 = attitude_estimation_complementary_filter(euler_from_accelerations1, euler_from_gyro1)
euler_from_madgwick1,roll_madgwick1, pitch_madgwick1, yaw_madgwick1 = madgwick_filter_function(imu_converted_vals1, imu_ts1, beta=0.01)

roll_vicon1, pitch_vicon1, yaw_vicon1 = vicon_to_rpy(vicon_rots1)

imu_timestamp1 = (np.array([imu_ts1])-np.array(imu_ts1[0])).flatten()
vicon_timestamp1 = (np.array([vicon_ts1])-np.array(imu_ts1[0])).flatten()

fig, ax = plt.subplots(3, 1, figsize=(16, 12))

ax[0].plot(imu_timestamp1, np.rad2deg(roll_gyro1), label="Gyroscope")
ax[0].plot(imu_timestamp1, np.rad2deg(roll_acc1), label="Accelerometer")
ax[0].plot(imu_timestamp1, np.rad2deg(roll_comp1), label="Complementary")
ax[0].plot(imu_timestamp1, np.rad2deg(roll_madgwick1), label="Madgwick")
ax[0].plot(vicon_timestamp1, np.rad2deg(roll_vicon1), label="Vicon")
ax[0].set_title("ROLL(X-axis)")
ax[0].set_ylabel('Angle(Degrees)')
ax[0].grid()
ax[0].legend()

ax[1].plot(imu_timestamp1, np.rad2deg(pitch_gyro1), label="Gyroscope")
ax[1].plot(imu_timestamp1, np.rad2deg(pitch_acc1), label="Accelerometer")
ax[1].plot(imu_timestamp1, np.rad2deg(pitch_comp1), label="Complementary")
ax[1].plot(imu_timestamp1, np.rad2deg(pitch_madgwick1), label="Madgwick")
ax[1].plot(vicon_timestamp1, np.rad2deg(pitch_vicon1), label="Vicon")
ax[1].set_title("PITCH(Y-axis)")
ax[1].set_ylabel('Angle(Degrees)')
ax[1].grid()
ax[1].legend()

ax[2].plot(imu_timestamp1, np.rad2deg(yaw_gyro1), label="Gyroscope")
ax[2].plot(imu_timestamp1, np.rad2deg(yaw_acc1), label="Accelerometer")
ax[2].plot(imu_timestamp1, np.rad2deg(yaw_comp1), label="Complementary")
ax[2].plot(imu_timestamp1, np.rad2deg(yaw_madgwick1), label="Madgwick")
ax[2].plot(vicon_timestamp1, np.rad2deg(yaw_vicon1), label="Vicon")
ax[2].set_title("YAW(Z-axis)")
ax[2].set_ylabel('Angle(Degrees)')
ax[2].set_xlabel('Time(s)')
ax[2].grid()
ax[2].legend()

plt.show()

##########################################################
## CODE FOR EXTRACTING VIDEO FROM ANIMATION OF ESTIMATES
##########################################################

a_rotation = np.zeros((3,3,euler_from_accelerations1.shape[0]))
for i in range(0,euler_from_accelerations1.shape[0]):
    a_rotation[:,:,i] = euler_to_rotation_matrix(euler_from_accelerations1[i,0], euler_from_accelerations1[i,1],euler_from_accelerations1[i,2])

gyro_rotation = np.zeros((3,3,euler_from_gyro1.shape[0]))
for i in range(0,euler_from_gyro1.shape[0]):
    gyro_rotation[:,:,i] = euler_to_rotation_matrix(euler_from_gyro1[i,0], euler_from_gyro1[i,1],euler_from_gyro1[i,2])

COM_rotation = np.zeros((3,3,euler_from_comp1.shape[1]))
for i in range(0,euler_from_comp1.shape[1]):
    COM_rotation[:,:,i] = euler_to_rotation_matrix(euler_from_comp1[0,i], euler_from_comp1[1,i],euler_from_comp1[2,i])

MAD_rotation = np.zeros((3,3,euler_from_madgwick1.shape[0]))
for i in range(0,euler_from_madgwick1.shape[0]):
    MAD_rotation[:,:,i] = euler_to_rotation_matrix(euler_from_madgwick1[i,0], euler_from_madgwick1[i,1],euler_from_madgwick1[i,2])
Framerate = 10 #fps

AnimFig=plt.figure()
FFMpegWriter = animation.writers['ffmpeg']

metadata = dict(title='', artist='Ankit Mittal', comment='')
writer = FFMpegWriter(fps=Framerate, metadata=metadata)

ViconTs = np.linspace(vicon_timestamp1[0], vicon_timestamp1[-1], num=(int(vicon_timestamp1[-1]-vicon_timestamp1[0])*Framerate) )
print('The dimensions of ViconTs are: ' + str(ViconTs.shape))
counter=0
Vicon_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
for i in range(vicon_rots1.shape[2]):
	#if you passed it grab the last one, should be close enough
	if vicon_timestamp1[i]>ViconTs[counter]:
		Vicon_Rot_Ms[:,:,counter]=vicon_rots1[:,:,i]
		counter+=1
	if counter==ViconTs.shape[0]:
		break

Accel_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
Gyro_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
COM_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
MAD_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))


counter=0
for i in range(0,euler_from_gyro1.shape[0]):

  if imu_timestamp1[i]>ViconTs[counter]:

    Accel_Rot_Ms[:,:,counter]=a_rotation[:,:,i]
    Gyro_Rot_Ms[:,:,counter]=gyro_rotation[:,:,i]
    COM_Rot_Ms[:,:,counter]=COM_rotation[:,:,i]
    MAD_Rot_Ms[:,:,counter]=MAD_rotation[:,:,i]
    counter+=1

  if counter==ViconTs.shape[0]:
    break



Fig_animate= plt.figure(figsize=(15,5))
ax1=plt.subplot(151,projection='3d')
ax2=plt.subplot(152,projection='3d')
ax3=plt.subplot(153,projection='3d')
ax4=plt.subplot(154,projection='3d')
ax5=plt.subplot(155,projection='3d')
ax1.title.set_text('Gyro')
ax2.title.set_text('Accel')
ax3.title.set_text('COM')
ax4.title.set_text('Vicon')
ax5.title.set_text('Madgwick')

print('writing file...')
filename = "V_Dataset6"
with writer.saving(Fig_animate, filename +"_vid"+ ".mp4", ViconTs.shape[0]):
	for i in range(ViconTs.shape[0]):
		print(str(i) +'of' + str(ViconTs.shape[0]))

		ax1.clear()
		ax2.clear()
		ax3.clear()
		ax4.clear()
		ax5.clear()
		ax1.title.set_text('Gyro')
		ax2.title.set_text('Accel')
		ax3.title.set_text('COM')
		ax4.title.set_text('Vicon')
		ax5.title.set_text('Madgwick')
		rotplot(Gyro_Rot_Ms[:,:,i],ax1)
		rotplot(Accel_Rot_Ms[:,:,i],ax2)
		rotplot(COM_Rot_Ms[:,:,i],ax3)
		rotplot(Vicon_Rot_Ms[:,:,i],ax4)
		rotplot(MAD_Rot_Ms[:,:,i],ax5)
		writer.grab_frame()