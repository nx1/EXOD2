from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import PickEvent
from matplotlib.colors import LogNorm
from astropy.table import Table

class EventListViewer:
    def __init__(self, tab):
        xmin, xmax = np.min(tab['X']), np.max(tab['X'])
        ymin, ymax = np.min(tab['Y']), np.max(tab['Y']) 
    
        nbins = 300
        xbins = np.linspace(xmin, xmax, nbins)
        ybins = np.linspace(ymin, ymax, nbins)
        
        img = plt.hist2d(x=tab['X'], y=tab['Y'], bins=(xbins,ybins))[0]
        plt.close()

        self.tab = tab
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor('black')
        self.image = img 
        self.img_plot = self.ax.imshow(self.image, aspect='equal', cmap='hot', origin='lower',
                                       norm=LogNorm(), interpolation='none',
                                       extent=[xmin,xmax,ymin,ymax])
        self.rect_size = 1000
        self.rect = Rectangle((0,0), self.rect_size, self.rect_size, edgecolor='cyan', linewidth=1, fill=False)
        self.ax.add_patch(self.rect)
        self.cursor_position = (0, 0)

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        x, y = self.cursor_position
        x_size = self.rect.get_width()
        y_size = self.rect.get_height()
        xlo, xhi = x-x_size, x+x_size
        ylo, yhi = y-y_size, y+y_size
        tab = self.tab
        tab_filt = tab[(tab['X'] > xlo) & (tab['X'] < xhi) &
                       (tab['Y'] > ylo) & (tab['Y'] < yhi)]
        print(tab_filt)
        print(xlo, xhi, ylo, yhi)
        self.plot_subset_image(tab_filt)

    def plot_subset_image(self, tab_filt):
        # Plot the highlighted subset in raw detector coordinates.
        tab_filt = tab_filt[tab_filt['CCDNR'] == mode(tab_filt['CCDNR'])]
        xbins = np.arange(tab_filt['RAWX'].min(), tab_filt['RAWX'].max(), 1)
        ybins = np.arange(tab_filt['RAWY'].min(), tab_filt['RAWY'].max(), 1)
        im, _, _ = np.histogram2d(x=tab_filt['RAWX'], y=tab_filt['RAWY'], bins=[xbins,ybins])
        plt.figure()
        plt.imshow(im, aspect='equal', cmap='hot', interpolation='none')
        plt.show()
    
    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            self.cursor_position = (x, y)
            self.update_square()

    def on_scroll(self, event):
        if event.inaxes==self.ax:
            delta = 100*event.step
            self.rect_size += delta
            self.rect_size = max(1, self.rect_size)
            self.update_square()

    def update_square(self):
        x, y = self.cursor_position
        xsize = x - self.rect_size / 2
        ysize = y - self.rect_size / 2

        self.rect.set_x(xsize)
        self.rect.set_y(ysize)
        self.rect.set_width(self.rect_size)
        self.rect.set_height(self.rect_size)
        self.fig.canvas.draw()

    def show(self):
        plt.show()

if __name__ == "__main__":
    fits_file = './PN_clean.fits'
    tab = Table.read(fits_file, hdu=1)
    print(tab)
    evt_viewer = EventListViewer(tab)
    evt_viewer.show()


