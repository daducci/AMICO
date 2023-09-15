import configparser
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('section', type=str, help='Section name')
parser.add_argument('library', type=str, help='Library name')
parser.add_argument('library_dir', type=str, help='Library directory path')
parser.add_argument('include_dir', type=str, help='Include directory path')
args = parser.parse_args()

# write the site.cfg file
config = configparser.ConfigParser()
config[args.section] = {
    'library': args.library,
    'library_dir': args.library_dir,
    'include_dir': args.include_dir
}
with open('site.cfg', 'w') as configfile:
    config.write(configfile)
