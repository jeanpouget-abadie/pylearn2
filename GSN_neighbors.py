!obj:pylearn2.train.Train {
  dataset: &X !obj:pylearn2.datasets.moons.Moons {
    num_X: 500,
    noise: 0.01
  },
  model: !obj:pylearn2.models.autoencoder.DivergenceGSN {
    act_enc: !obj:pylearn2.models.mlp.MLP {
      layers: [
        !obj:pylearn2.models.mlp.Tanh {
          layer_name: 'enc_tanh_1',
          dim: 20,
          irange: 0.01,
        },
        !obj:pylearn2.models.mlp.Linear {
          layer_name: 'enc_tanh_2',
          dim: 16,
          irange: 0.01,
        }
      ],
      nvis: 2,
    },
    act_dec: !obj:pylearn2.models.mlp.MLP {
      layers: [
        !obj:pylearn2.models.mlp.Tanh {
          layer_name: 'enc_tanh_1',
          dim: 16,
          irange: 0.01,
        },
        !obj:pylearn2.models.mlp.Linear {
          layer_name: 'enc_tanh_2',
          dim: 2,
          irange: 0.01,
        }
      ],
      nvis: 16,
    },
    corruptor: !obj:pylearn2.corruption.TrainableGaussianCorruptor {
      stdev: 1.0,
    },
    decorruptor: !obj:pylearn2.corruption.TrainableGaussianCorruptor {
      stdev: 0.5,
    }
  },
  algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
    learning_rate: 0.001,
    batch_size: 50,
    monitoring_dataset: {
      train: *X,
      valid: !obj:pylearn2.datasets.moons.Moons {
        num_X: 100,
        noise: 0.01
      }
    }, cost: !obj:pylearn2.costs.autoencoder.DivergenceCost {
      X: *X,
      num_samples: 200,
      num_encodings: 1,
    }
  },
  extensions: [
    !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
      channel_name: 'valid_cost',
      save_path: '${PYLEARN2_TRAIN_FILE_FULL_STEM}_best_cost.pkl'
    }, !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
      channel_name: 'valid_dec_stdev',
      save_path: '${PYLEARN2_TRAIN_FILE_FULL_STEM}_best_dec_stdev.pkl'
    }
  ],
  save_path: '${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl',
  save_freq: 100,
}
