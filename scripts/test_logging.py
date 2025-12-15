import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import logging_helper as lh
class DummyCFG:
    def get(self, s, k):
        return 'INFO'

def cb(msg):
    print('CALLBACK:', msg[:60])

print('Running normally (not frozen):')
lh.emit(cb, DummyCFG(), 'INFO', 'Test message normal')
lh.emit(None, DummyCFG(), 'INFO', 'Test message print')

print('\nSimulating frozen:')
setattr(sys, 'frozen', True)
lh.emit(cb, DummyCFG(), 'INFO', 'Test message frozen with cb')
lh.emit(None, DummyCFG(), 'INFO', 'Test message frozen without cb')
