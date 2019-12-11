from log_parsers import Drain

input_dir = '../logs/HDFS/'
output_dir = '../parsing_result/'
log_file = 'HDFS.log'
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

regex = [r'blk_(|-)[0-9]+', r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'] # block ID and IP
st = 0.5
depth = 3

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=True)
parser.parse(log_file)