import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='', type=str, help='Root path of input videos')
    parser.add_argument('--model', default='', type=str, help='Model file path')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--file_list_mode', action='store_true', help='If file list mode is used')
    parser.add_argument("--file_list", type=str, default='')
    parser.add_argument('--no_norm', action='store_true', help='Features should not be normalized to 32 features.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args

class create_args():
    def __init__(self, input='input', video_root='./videos', model='', output='output.json',\
                 mode='feature', batch_size=32, n_threads=4, model_name='resnet', model_depth=101,\
                 resnet_shortcut='A', wide_resnet_k=2, resnext_cardinality=32, no_cuda=False, \
                 verbose=False, n_classes=10, sample_size=112, sample_duration=16):
        self.input = input
        self.video_root = video_root
        self.model = model
        self.output = output
        self.mode = mode
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.model_name = model_name
        self.model_depth = model_depth
        self.resnet_shortcut = resnet_shortcut
        self.wide_resnet_k = wide_resnet_k
        self.resnext_cardinality = resnext_cardinality
        self.no_cuda = no_cuda
        self.verbose = verbose
        self.n_classes = n_classes
        self.sample_size=112
        self.sample_duration=16
