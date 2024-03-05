#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from openpyxl import load_workbook
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')

def load_dataset():
  voltage = load_workbook(join(FLAGS.input_dir, 'Voltage.xlsx')); voltage = voltage.active
  current = load_workbook(join(FLAGS.input_dir, 'Current.xlsx')); current = current.active
  real = load_workbook(join(FLAGS.input_dir, 'EIS_real.xlsx')); real = real.active
  imag = load_workbook(join(FLAGS.input_dir, 'EIS_imag.xlsx')); imag = imag.active
  samples = list()
  for col in range(voltage.max_column):
    v, c = list(), list()
    for row in range(voltage.max_row):
      v.append(voltage.cell(row = row + 1, column = col + 1).value)
      c.append(current.cell(row = row + 1, column = col + 1).value)
    v = np.array(v)
    c = np.array(c)
    pulse = np.stack([v,c], axis = -1) # pulse.shape = (length, 2)
    r, i = list(), list()
    for row in range(real.max_row):
      r.append(real.cell(row = row + 1, column = col + 1).value)
      i.append(imag.cell(row = row + 1, column = col + 1).value)
    r = np.array(r)
    i = np.array(i)
    eis = np.stack([r,i], axis = -1) # eis.shape = (length, 2)
    samples.append((pulse, eis))
  return samples

def main(unused_argv):
  samples = load_dataset()
  if exists(FLAGS.ckpt): rmtree(FLAGS.ckpt)
  mkdir(FLAGS.ckpt)
  x = np.stack([sample[0].flatten() for sample in samples], axis = 0) # x.shape = (sample_num, 1800*2)
  y = np.stack([sample[1].flatten() for sample in samples], axis = 0) # y.shape = (sample_num, 35*2)
  models = [SVR(C=1.0, epsilon = 0.2) for i in range(35*2)]
  for i in range(35 * 2):
    regr = make_pipeline(StandardScaler(), models[i])
    regr.fit(x, y[:,i])
    with open(join(FLAGS.ckpt, '%d.pickle' % i), 'wb') as f:
      f.write(pickle.dumps(models[i]))

if __name__ == "__main__":
  add_options()
  app.run(main)

