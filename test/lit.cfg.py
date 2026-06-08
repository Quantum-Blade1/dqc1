import lit.formats
import os

config.name = 'DQC'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
config.substitutions.append(('%PATH%', config.environment.get('PATH', '')))
llvm_bin = '/opt/homebrew/opt/llvm/bin'
config.environment['PATH'] = os.path.join(build_dir, 'tools', 'dqc-opt') + os.pathsep + llvm_bin + os.pathsep + config.environment.get('PATH', '')
