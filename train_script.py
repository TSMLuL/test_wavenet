import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from scipy.io import wavfile

import matplotlib as mpl
import matplotlib.pyplot as plt

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = WaveNetModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     dtype=dtype,
                     bias=True)

#model = load_latest_model_from('snapshots', use_cuda=True)
#model = torch.load('snapshots/some_model')

if use_cuda:
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=500)

print('the dataset has ' + str(len(data)) + ' items')


def generate_and_log_samples(step):

    sample_length=32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('st_{}_temperature_0.5'.format(str(step)), tf_samples, step, sr=sample_length/4)

    out_file = 'generated_samples/haconne_temp_0.5_str_{}.wav'.format(str(step))
    np_sample = np.asarray(samples)
    wavfile.write(out_file, 11025, np.clip(np_sample[0], a_min=-1.0, a_max=1.0))

    out_file_png = 'generated_samples/haconne_temp_0.5_str_{}.png'.format(str(step))

    print(np_sample[0])

    # set plotting appearance
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [10, 10]  # width * height
    plt.rcParams['agg.path.chunksize'] = 1000000
    fig, ax = plt.subplots(1, 1)

    ax.plot(np_sample[0])

    # set grid and tight plotting layout
    plt.grid(True)
    plt.tight_layout()

    # save plot to plotting directory
    plt.savefig(out_file_png, dpi=300)

    # close plot
    plt.close()

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    #logger.audio_summary('st_{}_temperature_1.0'.format(str(step)), tf_samples, step, sr=sample_length/4)

    out_file = 'generated_samples/haconne_temp_1.0_st_{}.wav'.format(str(step))
    np_sample = np.asarray(samples)
    wavfile.write(out_file, 11025, np.clip(np_sample[0], a_min=-1.0, a_max=1.0))

    out_file_png = 'generated_samples/haconne_temp_1.0_st_{}.png'.format(str(step))

    # set plotting appearance
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [10, 10]  # width * height
    plt.rcParams['agg.path.chunksize'] = 1000000
    fig, ax = plt.subplots(1, 1)

    ax.plot(np_sample[0])

    # set grid and tight plotting layout
    plt.grid(True)
    plt.tight_layout()

    # save plot to plotting directory
    plt.savefig(out_file_png, dpi=300)

    # close plot
    plt.close()

    print("audio clips generated")

logger = TensorboardLogger(log_interval=10,
                           validation_interval=10,
                           generate_interval=10,
                           generate_function=generate_and_log_samples,
                           log_dir="logs/chaconne_model")

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=10,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

#out_file = 'generated_samples/haconne_temp_1.0_st_{}.wav'.format(str(100))
#np_sample = np.asarray(np.random.uniform(-1.0, 1.0, 1600))
#wavfile.write(out_file, 160, np_sample)

print('start training...')
trainer.train(batch_size=16, epochs=100, continue_training_at_step=0)

#added this accelerator on 03/04/2019

start_data = data[250000][0] # use start data from the data set
start_data = torch.max(start_data, 0)[1] # convert one hot vectors to integers

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

generated = model.generate_fast(num_samples=160000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=1000,
                                 temperature=1.0,
                                 regularize=0.)
