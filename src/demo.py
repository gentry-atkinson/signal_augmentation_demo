#Author: Gentry Atkinson
#Organization: Texas University
#Data: 5 May, 2022
#Demonstrate signal augmentation

from gen_ts_data import  generate_pattern_data_as_array
from ts_feature_toolkit import get_features_from_one_signal
import augmentation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

if __name__ == '__main__':
    signalLength = 150
    originalSignal = np.array(generate_pattern_data_as_array(signalLength))
    secondSignal = np.array(generate_pattern_data_as_array(signalLength))
    plt.plot(range(signalLength), originalSignal)
    plt.title('Original')
    plt.savefig('imgs/original_signal.pdf')

    flip = np.array(augmentation.flip(originalSignal))
    randomSignalDrop = np.array(augmentation.rand_signal_drop(originalSignal))
    windowedSignalDrop = np.array(augmentation.wind_signal_drop(originalSignal))
    fixedSignalDrop = np.array(augmentation.fixed_signal_drop(originalSignal))
    gaussianNoiseInjection = np.array(augmentation.gau_noise_inj(originalSignal))
    amplitudeShift = np.array(augmentation.amp_shift(originalSignal))
    timeShift = np.array(augmentation.time_shift(originalSignal))

    feature_dict = {
        'original' : get_features_from_one_signal(originalSignal),
        'second' : get_features_from_one_signal(secondSignal),
        'flip' : get_features_from_one_signal(flip),
        'randomSignalDrop' : get_features_from_one_signal(randomSignalDrop),
        'windowedSignalDrop' : get_features_from_one_signal(windowedSignalDrop),
        'fixedSignalDrop' : get_features_from_one_signal(fixedSignalDrop),
        'gaussianNoiseInjection' : get_features_from_one_signal(gaussianNoiseInjection),
        'amplitudeShift' : get_features_from_one_signal(amplitudeShift),
        'timeShift' : get_features_from_one_signal(timeShift),
    }

    mse_dict = {
        'original' : np.mean([(originalSignal[i] - originalSignal[i])**2 for i in range(signalLength)]),
        'second' : np.mean([(originalSignal[i] - secondSignal[i])**2 for i in range(signalLength)]),
        'flip' : np.mean([(originalSignal[i] - flip[i])**2 for i in range(signalLength)]),
        'randomSignalDrop' : np.mean([(originalSignal[i] - randomSignalDrop[i])**2 for i in range(signalLength)]),
        'windowedSignalDrop' : np.mean([(originalSignal[i] - windowedSignalDrop[i])**2 for i in range(signalLength)]),
        'fixedSignalDrop' : np.mean([(originalSignal[i] - fixedSignalDrop[i])**2 for i in range(signalLength)]),
        'gaussianNoiseInjection' : np.mean([(originalSignal[i] - gaussianNoiseInjection[i])**2 for i in range(signalLength)]),
        'amplitudeShift' : np.mean([(originalSignal[i] - amplitudeShift[i])**2 for i in range(signalLength)]),
        'timeShift' : np.mean([(originalSignal[i] - timeShift[i])**2 for i in range(signalLength)]),
    }

    print('---Mean Squared Error---')
    print('Original to Original: {}'.format(mse_dict['original']))
    print('Original to Second Signal: {}'.format(mse_dict['second']))
    print('Original to Flipped Signal: {}'.format(mse_dict['flip']))
    print('Original to Random Signal Drop: {}'.format(mse_dict['randomSignalDrop']))
    print('Original to Windowed Signal Drop: {}'.format(mse_dict['windowedSignalDrop']))
    print('Original to Periodic Signal Drop: {}'.format(mse_dict['fixedSignalDrop']))
    print('Original to Noise Injection: {}'.format(mse_dict['gaussianNoiseInjection']))
    print('Original to Shifted Amplitude: {}'.format(mse_dict['amplitudeShift']))
    print('Original to Shifted Time Scale: {}'.format(mse_dict['timeShift']))

    print('\n---Cosine Distances---')
    print('Original to Original: {}'.format(distance.cosine(feature_dict['original'], feature_dict['original'])))
    print('Original to Second Signal: {}'.format(distance.cosine(feature_dict['original'], feature_dict['second'])))
    print('Original to Flipped Signal: {}'.format(distance.cosine(feature_dict['original'], feature_dict['flip'])))
    print('Original to Random Signal Drop: {}'.format(distance.cosine(feature_dict['original'], feature_dict['randomSignalDrop'])))
    print('Original to Windowed Signal Drop: {}'.format(distance.cosine(feature_dict['original'], feature_dict['windowedSignalDrop'])))
    print('Original to Periodic Signal Drop: {}'.format(distance.cosine(feature_dict['original'], feature_dict['fixedSignalDrop'])))
    print('Original to Noise Injection: {}'.format(distance.cosine(feature_dict['original'], feature_dict['gaussianNoiseInjection'])))
    print('Original to Shifted Amplitude: {}'.format(distance.cosine(feature_dict['original'], feature_dict['amplitudeShift'])))
    print('Original to Shifted Time Scale: {}'.format(distance.cosine(feature_dict['original'], feature_dict['timeShift'])))

    #Flip
    plt.figure()
    plt.plot(range(signalLength), flip)
    plt.title('Flip')
    plt.savefig('imgs/flip.pdf')

    #Random Signal Drop
    plt.figure()
    plt.plot(range(signalLength), randomSignalDrop)
    plt.title('Random Signal Drop')
    plt.savefig('imgs/rand_signal_drop.pdf')

    #Windowed Signal Drop
    plt.figure()
    plt.plot(range(signalLength), windowedSignalDrop)
    plt.title('Windowed Signal Drop')
    plt.savefig('imgs/wind_signal_drop.pdf')

    #Fixed Signal Drop
    plt.figure()
    plt.plot(range(signalLength), fixedSignalDrop)
    plt.title('Periodic Signal Drop')
    plt.savefig('imgs/fixed_signal_drop.pdf')
    
    #Gaussian Noise Injection
    plt.figure()
    plt.plot(range(signalLength), gaussianNoiseInjection)
    plt.title('Noise Injection')
    plt.savefig('imgs/gaussina_noise_injection.pdf')

    #Amplitude Shift
    plt.figure()
    plt.plot(range(signalLength), amplitudeShift)
    plt.title('Amplitude Shift')
    plt.savefig('imgs/amplitude_shift.pdf')

    #Time Shift
    plt.figure()
    plt.plot(range(signalLength), timeShift)
    plt.title('Time Shift')
    plt.savefig('imgs/time_shift.pdf')

    
    