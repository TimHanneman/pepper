import onnxruntime
import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pepper.modules.python.models.dataloader_predict import SequenceDataset
from tqdm import tqdm
from pepper.modules.python.models.ModelHander import ModelHandler
from pepper.modules.python.Options import ImageSizeOptions, TrainOptions
from pepper.modules.python.DataStorePredict import DataStore
import torch.onnx
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


def predict(input_filepath, file_chunks, output_filepath, batch_size, num_workers, rank, threads, model_path):
    # session options
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession(model_path + ".onnx", sess_options=sess_options)
    torch.set_num_threads(threads)

    # create output file
    output_filename = output_filepath + "pepper_prediction_" + str(rank) + ".hdf"
    prediction_data_file = DataStore(output_filename, mode='w')

    # data loader
    input_data = SequenceDataset(input_filepath, file_chunks)
    data_loader = DataLoader(input_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    if rank == 0:
        progress_bar = tqdm(
            total=len(data_loader),
            ncols=100,
            leave=False,
            position=rank,
            desc="CALLER #" + str(rank),
        )

    with torch.no_grad():
        for contig, contig_start, contig_end, chunk_id, images, position, index in data_loader:
            images = images.type(torch.FloatTensor)

            hidden = torch.zeros(images.size(0), 2 * TrainOptions.LSTM_LAYERS, TrainOptions.HIDDEN_SIZE)
            #
            cell_state = torch.zeros(images.size(0), 2 * TrainOptions.LSTM_LAYERS, TrainOptions.HIDDEN_SIZE)

            prediction_base_tensor = torch.zeros((images.size(0), images.size(1), ImageSizeOptions.TOTAL_LABELS))

            for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                    break
                chunk_start = i
                chunk_end = i + TrainOptions.TRAIN_WINDOW
                # chunk all the data
                image_chunk = images[:, chunk_start:chunk_end]

                # run inference on onnx mode, which takes numpy inputs
                #
                ##This looks into this section
                ort_inputs = {ort_session.get_inputs()[0].name: image_chunk.cpu().numpy(),
                              ort_session.get_inputs()[1].name: hidden.cpu().numpy(),
                              ort_session.get_inputs()[2].name: cell_state.cpu().numpy()}
                #
                #output_base, hidden, cell_state = ort_session.run(None, ort_inputs)
                output_base, hidden, cell_state = ort_session.run(None, ort_inputs)
                output_base = torch.from_numpy(output_base)
                hidden = torch.from_numpy(hidden)
                #
                cell_state = torch.from_numpy(cell_state)

                # now calculate how much padding is on the top and bottom of this chunk so we can do a simple
                # add operation
                top_zeros = chunk_start
                bottom_zeros = ImageSizeOptions.SEQ_LENGTH - chunk_end

                counts = torch.ones((output_base.size(0), output_base.size(1), 1))

                # do softmax and get prediction
                # we run a softmax a padding to make the output tensor compatible for adding
                inference_layers = nn.Sequential(
                    nn.Softmax(dim=2),
                    nn.ZeroPad2d((0, 0, top_zeros, bottom_zeros))
                )
                base_prediction = inference_layers(output_base)

                # now simply add the tensor to the global counter
                prediction_base_tensor = torch.add(prediction_base_tensor, base_prediction)

            base_values, base_labels = torch.max(prediction_base_tensor, 2)

            # this part is for the phred score calculation
            counts = torch.ones((base_values.size(0), base_values.size(1) - 2 * ImageSizeOptions.SEQ_OVERLAP))
            top_ones = nn.ZeroPad2d((ImageSizeOptions.SEQ_OVERLAP, ImageSizeOptions.SEQ_OVERLAP))
            counts = top_ones(counts) + 1
            phred_score = -10 * torch.log10(1.0 - (base_values / counts))
            phred_score[phred_score == float('inf')] = 100

            predicted_base_labels = base_labels.cpu().numpy()
            phred_score = phred_score.cpu().numpy()

            for i in range(images.size(0)):
                prediction_data_file.write_prediction(contig[i], contig_start[i], contig_end[i], chunk_id[i],
                                                      position[i], index[i], predicted_base_labels[i], phred_score[i])

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()


def cleanup():
    dist.destroy_process_group()


def setup(rank, total_callers, args, all_input_files):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=total_callers)

    filepath, output_filepath, model_path, batch_size, num_workers, threads = args

    # issue with semaphore lock: https://github.com/pytorch/pytorch/issues/2517
    # mp.set_start_method('spawn')

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases. https://github.com/pytorch/pytorch/issues/2517
    # torch.manual_seed(42)
    predict(filepath,
            all_input_files[rank],
            output_filepath,
            batch_size,
            num_workers,
            rank,
            threads,
            model_path)
    cleanup()


def predict_distributed_cpu(filepath, file_chunks, output_filepath, model_path, batch_size, total_callers, threads,
                            num_workers):
    """
    Create a prediction table/dictionary of an images set using a trained model.
    :param filepath: Path to image files to predict on
    :param file_chunks: Path to chunked files
    :param batch_size: Batch size used for prediction
    :param model_path: Path to a trained model
    :param output_filepath: Path to output directory
    :param total_callers: Number of callers to spawn
    :param threads: Number of threads to use per caller
    :param num_workers: Number of workers to be used by the dataloader
    :return: Prediction dictionary
    """
    # load the model and create an ONNX session
    transducer_model, hidden_size, gru_layers, prev_ite = \
        ModelHandler.load_simple_model_for_training(model_path,
                                                    input_channels=ImageSizeOptions.IMAGE_CHANNELS,
                                                    image_features=ImageSizeOptions.IMAGE_HEIGHT,
                                                    seq_len=ImageSizeOptions.SEQ_LENGTH,
                                                    num_classes=ImageSizeOptions.TOTAL_LABELS)
    transducer_model.eval()

    sys.stderr.write("INFO: MODEL LOADING TO ONNX\n")
    x = torch.zeros(1, TrainOptions.TRAIN_WINDOW, ImageSizeOptions.IMAGE_HEIGHT)
    h = torch.zeros(1, 2 * TrainOptions.LSTM_LAYERS, TrainOptions.HIDDEN_SIZE)
    #
    ce = torch.zeros(1, 2 * TrainOptions.LSTM_LAYERS, TrainOptions.HIDDEN_SIZE)


    if not os.path.isfile(model_path + ".onnx"):
        sys.stderr.write("INFO: SAVING MODEL TO ONNX\n")
        #
        #torch.onnx.export(transducer_model, (x, h),
        torch.onnx.export(transducer_model, (x, h, ce),
                          model_path + ".onnx",
                          training=False,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input_image', 'input_hidden'],
                          output_names=['output_pred', 'output_hidden'],
                          dynamic_axes={'input_image': {0: 'batch_size'},
                                        'input_hidden': {0: 'batch_size'},
                                        'output_pred': {0: 'batch_size'},
                                        'output_hidden': {0: 'batch_size'}})

    args = (filepath, output_filepath, model_path, batch_size, num_workers, threads)
    mp.spawn(setup,
             args=(total_callers, args, file_chunks),
             nprocs=total_callers,
             join=True)
