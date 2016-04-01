from os import fstat
from sys import stdout
from math import ceil

class ProgressBar :
    """A simple text-based progress indicator."""

    def __init__( self, n, width = 50, prefix = "", erase=False ) :
        """Initialize the progress indicator.

        Parameters
        ----------
        n : integer
            Total number of steps to track
        width : integer
            Width (in characthers) of the bar indicator (default : 50)
        prefix : string
            A string to prepend to the indicator bar (default : "")
        erase : boolean
            True -> erase the bar when done, False -> simply go to new line (default : False)
        """
        self.i      = 1
        self.n      = n
        self.width  = width
        self.nbars  = 0
        self.prefix = prefix
        self.erase  = erase
        if fstat(0)==fstat(1): #only show progress bar if stdout, False if redirected
            print '\r%s[%s] %5.1f%%' % ( prefix, ' ' * width, 0.0 ),
            stdout.flush()


    def update( self ) :
        """Update the status and redraw the progress bar (if needed)."""
        if self.i < 1 or self.i > self.n :
            return
        ratio = float(self.i) / float(self.n)
        nbars = int( ceil(ratio * self.width) )
        if self.i == 1 or self.i == self.n or self.i or nbars != self.nbars:
            self.nbars = nbars
            if fstat(0)==fstat(1): #only show progress bar if stdout
                print '\r%s[%s%s] %5.1f%%' % ( self.prefix, '=' * self.nbars, ' ' * (self.width-self.nbars), 100.0*ratio),
                stdout.flush()
        self.i += 1

        if self.i > self.n :
            if self.erase :
                print '\r' + ' ' * ( len(self.prefix) + self.width + 9 ) + '\r',
            else:
                print
            stdout.flush()
