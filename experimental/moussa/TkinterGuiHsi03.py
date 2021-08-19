# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:51:08 2021

@author: mzomou

"""
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import IntVar, DISABLED, ACTIVE, NORMAL, StringVar, messagebox 
from tkinter.colorchooser import askcolor
import numpy as np
from hsi import HSAbsorption, HSImage
from hsi.analysis import HSComponentFit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from tkinter import filedialog
from scipy import ndimage
from matplotlib.widgets import Cursor
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tkinter import font as tkFont
from tkinter import simpledialog
from spectral import spectral_angles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import susi

##### Cube-Data Laden und als RGB Bild Darstellen
WLD = np.linspace(500, 995, 100)
I = 0
FilterState = ''
LoadState = ''

def LoadCubeData ():
    
    global I, ax, ax2, GraphFrame, cubeData, rgb_image, Abs, f, FilterState, LoadState, cursor, newFilter, NRows, NCols, NBands, toolbar, SegState, Absorption, Intensity, ResAbs, ResInt
    
    ##### Darstellen Aktualisieren
    
    if I > 0 :
        ax.clear()
        ax2.clear()
        
        GraphFrame.place_forget()
        GraphFrame = tk.Frame(root, bg='gray')
        GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)
        
        buttonS.config(state=ACTIVE)
        SegState = False
    
    ##### Cube-Data Auswählen
    
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    dim=3
    dtype=np.float32
    size = np.dtype(dtype).itemsize
    
    ##### Cube-Data Entpacken

    with open(filename,'rb') as file:
        dtypeHeader = np.dtype(np.int32)
        dtypeHeader = dtypeHeader.newbyteorder('>')
        buffer = file.read(size*dim)
        header = np.frombuffer(buffer, dtype=dtypeHeader)
       
        dtypeData = np.dtype(dtype)
        dtypeData = dtypeData.newbyteorder('>')
        buffer = file.read()
        cubeData = np.frombuffer(buffer, dtype=dtypeData).copy()
        
        cubeData = cubeData.reshape(header, order='C')
        cubeData = np.rot90(cubeData)
        
        LoadState = 'Image Loaded'
        
    cubeData [cubeData < 0.00001]= 0.00001
    
    imgSize    = np.shape(cubeData)
    NRows      = imgSize[0]
    NCols      = imgSize[1]
    NBands     = imgSize[2]
    
    ##### RGB Bild Erzeugen

    rval = cubeData[:,:,27]
    gval = cubeData[:,:,16]
    bval = cubeData[:,:,8]

    rgb = (np.stack([rval, gval, bval], axis = 2)).clip(0., 1.)

    ##### RGB mit gamma Korrektur

    scale = 1
    mid = 0.6
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img_gray = np.dot(rgb, rgb_weights)
    mean = np.mean(img_gray)
    gamma = np.log(mid) / np.log(mean)
    rgb_image = (scale * np.power(rgb, scale*gamma)).clip(0., 1.)
    
    ##### Residium
    
    hsImage = HSImage(filename)
    hsImage.set_format(HSAbsorption)
    hsImage.add_filter(mode='image', filter_type='mean', size=5)
    spectra = hsImage.fspectra
    wavelen = hsImage.wavelen
    analysis = HSComponentFit(hsformat=HSAbsorption)
    analysis.loadtxt("basevectors_3.txt", mode='all')
    analysis.set_data(spectra, wavelen, hsformat=HSAbsorption)
    analysis.set_roi([500e-9, 995e-9])
    analysis.set_var_bounds("hhb", [0, 0.05])
    analysis.set_var_bounds("ohb", [0, 0.05])
    analysis.set_var_bounds("wat", [0, 2.00])
    analysis.set_var_bounds("fat", [0, 1.00])
    analysis.set_var_bounds("mel", [0, 0.05])
    analysis.remove_component("mel")
    analysis.prepare_ls_problem()

    analysis.fit(method='bvls_f')

    # get residual
    
    a = analysis._anaSysMatrix # matrix of base vectors
    b = analysis._anaTrgVector # spectra to be fitted
    x = analysis._anaVarVector # vector of unknowns
    
    res = b - a[:, :-1] @ x[:-1,:]
    res = res.reshape(hsImage.shape)
    res_int = np.exp(-res)
        
    res = np.moveaxis (res, 0, -1)
    res_int = np.moveaxis (res_int, 0, -1)    
    
    ##### calculat Absorbtion
    
    Abs = -np.log(np.abs(cubeData))  
    
    FilterState = 'No Filter'
    label3.config(state = NORMAL)
    label3['text'] = 'Spectra without Filtering'
    newFilter = True
    
    ##### HS_Signals
    
    Absorption = Abs
    Intensity = cubeData
    ResAbs = res
    ResInt = res_int
    
    ##### Plot Results
    
    f = Figure()
    ax = f.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
    ax.imshow(rgb_image)
    ax2 = f.add_subplot(122, title = 'HS-Signal spectra', xlabel='Wave length')
    ax2.grid()
    # ax2.legend()
    
    canvas = FigureCanvasTkAgg(f, GraphFrame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    canvas._tkcanvas.pack()
    
    cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
    
    NavigationToolbar2Tk.toolitems = (
                                      ('Home', 'Reset original view', 'home', 'home'),
                                      ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                                      ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                                      ('Save', 'Save the figure', 'filesave', 'save_figure'),
                                     )
    
    toolbar = NavigationToolbar2Tk(canvas, GraphFrame)
    toolbar.update()
    
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    I += 1
    
    if LoadState == 'Image Loaded':
        
        selectHSSignal.config(state=ACTIVE)
        buttonS.config(state=ACTIVE)
        
##### Dialog Fenster Erzeugen

class OptionDialog(tk.Toplevel):
    """
        This dialog accepts a list of options.
        If an option is selected, the results property is to that option value
        If the box is closed, the results property is set to zero
    """
    def __init__(self,parent,title,question,options,size):
        tk.Toplevel.__init__(self,parent)
        self.title(title)
        self.question = question
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW",self.cancel)
        self.geometry(size)
        self.options = options
        self.result = '_'
        self.createWidgets()
        self.grab_set()
        ## wait.window ensures that calling function waits for the window to
        ## close before the result is returned.
        self.wait_window()
    def createWidgets(self):
        frmQuestion = tk.Frame(self)
        tk.Label(frmQuestion,text=self.question, font = ('Helvetica', '11')).grid()
        frmQuestion.grid(row=1 , ipady = 10, padx = 100, pady = 5)
        frmButtons = tk.Frame(self)
        frmButtons.grid(row=2, padx = 100)
        column = 0
        for option in self.options:
            btn = tk.Button(frmButtons,text=option, font = ('Helvetica', '11'), command=lambda x=option:self.setOption(x))
            btn.grid(column=column, row=1)
            column += 1 
    def setOption(self,optionSelected):
        self.result = optionSelected
        self.destroy()
    def cancel(self):
        self.result = None
        self.destroy()
        
##### Segmetierung

SegState = False

def Segmentation():
    
    global cubeData, rgb_image, ax, f, HSSig, FilterState, HSSigMean, HSSigMedian, HSSigSVN, SegState, Index, dlg, label_im, Absorption, Intensity, ResAbs, ResInt
    
    ABS =ndimage.uniform_filter(cubeData[:, :, 10], size=5) 
    REF = ndimage.uniform_filter(cubeData[:, :, 30], size=5)
            
    Index = np.divide(np.subtract(REF, ABS), np.add(REF, ABS))
        
    values = ['Thyroid gland','Another data']
    size = '500x100'
    dlg = OptionDialog(root,'TestDialog',"Which data do you want segmentat?",values, size)
    
    while dlg.result != "Thyroid gland" and dlg.result != "Another data":
        
        res = messagebox.showwarning('Segmentation', 'please select one data type')
        
        if res == None:
                    
            res = messagebox.showwarning('Segmentation', 'please select one data type')
                    
        else :
                    
            dlg = OptionDialog(root,'TestDialog',"Which data do you want segmentat?",values, size)
    
    if dlg.result == "Another data":
        
        # Index[Index>0.9]=0
        # Index[Index<0.1]=0
        # Index[Index>0]=1
        
        Index[Index>0.9]=0
        Index[Index<0.25]=0
        Index[Index>0]=1
    
        Index = ndimage.binary_fill_holes(Index)  

        Intensity = np.multiply(Intensity, Index[:,:,None])
        Absorption = np.multiply(Absorption, Index[:,:,None])
        ResInt = np.multiply(ResInt, Index[:,:,None])
        ResAbs = np.multiply(ResAbs, Index[:,:,None])
                
        rgb_image = np.multiply(rgb_image, Index[:,:,None])
        rgb_image[rgb_image==0]=1
        
        if FilterState == 'Mean Filter':
            HSSigMean = np.multiply(HSSigMean, Index[:,:,None])
        elif FilterState == 'Median Filter':
            HSSigMedian = np.multiply(HSSigMedian, Index[:,:,None])
        elif FilterState == 'SVN':
            HSSigSVN = np.multiply(HSSigSVN, Index[:,:,None])
        
    elif dlg.result == "Thyroid gland":
        
        Index[Index>1]=0
        Index[Index<0.5]=0
        Index[Index>0]=1     
        
        TG = Index*Abs[:,:,97]
        TGMean = ndimage.uniform_filter(TG, size=5)
        
        Index2 = np.exp(TGMean)
        Index2[Index2 <= 3.25] = 0
        Index2[Index2 > 3.25] = 1
        # Index2[Index2 <= 3.75] = 0
        # Index2[Index2 > 3.75] = 1
        Index2 = ndimage.binary_fill_holes(Index2)
        
        label_im, nb_labels = ndimage.label(Index2)
        sizes = ndimage.sum(Index2, label_im, range(nb_labels + 1))
        mask_size = sizes < 10000
        # mask_size = sizes < 4000
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
        label_im[label_im>1] = 1

        Intensity = np.multiply(Intensity, label_im[:,:,None])
        Absorption = np.multiply(Absorption, label_im[:,:,None])
        ResInt = np.multiply(ResInt, label_im[:,:,None])
        ResAbs = np.multiply(ResAbs, label_im[:,:,None]) 
                
        rgb_image = np.multiply(rgb_image, label_im[:,:,None])
        rgb_image[rgb_image==0]=1
        
        if FilterState == 'Mean Filter':
            HSSigMean = np.multiply(HSSigMean, label_im[:,:,None])
        elif FilterState == 'Median Filter':
            HSSigMedian = np.multiply(HSSigMedian, label_im[:,:,None])
        elif FilterState == 'SVN':
            HSSigSVN = np.multiply(HSSigSVN, label_im[:,:,None])
            
    Seg = ax.imshow(rgb_image)
    Seg.figure.canvas.draw()
    buttonS.config(state=DISABLED)
    SegState = True
    
##### HS-Signal auswählen
hssig = 0
ChangeHSSig = False
signame = ''

def SelectHsSignal(event):
    
    global Intensity, Absorption, ResInt, ResAbs, HSSig, hssig, ChangeHSSig, signame, newColor, FilterState, DrawState
    
    sig = False
    
    if varSig.get() == 'Intensity':
        
        HSSig = Intensity
        sig = True
        print(varSig.get())
        
    elif varSig.get() == 'Absorption':
        
        HSSig = Absorption
        sig = True
        print(varSig.get())
        
    elif varSig.get() == 'Residium from Intensity':
        
        HSSig = ResInt
        sig = True
        print(varSig.get())
        
    elif varSig.get() == 'Residium from Absorption':
        
        HSSig = ResAbs
        sig = True
        print(varSig.get())
        
    if sig == True and hssig < 1:
        
        checkbutton.config(state=ACTIVE)
        checkbutton2.config(state=ACTIVE)
        button3.config(state=ACTIVE)
        button9.config(state=ACTIVE)
        label3['text'] = varSig.get() + '\n' 'Spectra without Filtering'
    
    hssig += 1
    
    if hssig > 1 and signame != varSig.get():
        
        ChangeHSSig = True
        AddClearState.set(0)
        newColor = False
        FilterState = 'No Filter'
        label3['text'] = varSig.get() + '\n' 'Spectra without Filtering'
        
        if DrawState == True:
            
            button6.config(state=ACTIVE)

    else:
        
        ChangeHSSig = False
        
    signame = varSig.get()
        
##### Räumliche Filterung Auswählen
    
def CheckSpacial ():
    
    if SpacialFilter.get()==1:
        checkbutton2.config(state=DISABLED)
        rb1.config(state = ACTIVE)
        rb2.config(state = ACTIVE)
        label2.config(state = NORMAL)
        entry.config(state=NORMAL)
        
               
    elif SpacialFilter.get()==0:
        checkbutton2.config(state=ACTIVE)
        rb1.config(state = DISABLED)
        rb2.config(state = DISABLED)
        label2.config(state = DISABLED)
        entry.delete(0)
        entry.config(state=DISABLED)
        button2['state'] = DISABLED
        FilterChoice.set(0)

##### Räumlicher Filter Auswählen

def ShooseFilter():

    if (FilterChoice.get()==1 or FilterChoice.get()==2) and entry.get() != "" and not entry.get().isspace() and entry.get().isdigit():
       
        button2['state'] = ACTIVE
    else:
        button2['state'] = DISABLED
        
##### Filtergröße Kontrolle

def callback(sv):
    
    if (FilterChoice.get()==1 or FilterChoice.get()==2) and sv.get() != "" and not sv.get().isspace() and sv.get().isdigit():
        
        button2['state'] = ACTIVE
    else:
        button2['state'] = DISABLED
        
##### Filtergröße Kontrolle

def callback2(nbp):
    
    if nbp.get() != "" and not nbp.get().isspace() and nbp.get().isdigit():
        
        button2['state'] = ACTIVE
    else:
        button2['state'] = DISABLED
        
##### Spektrale Filterung Auswählen
        
def CheckSpectral ():
    
    if SpectralFilter.get()==1:
        checkbutton.config(state=DISABLED)
        #button2['state'] = ACTIVE
        entry2.config(state=NORMAL)
        
    elif SpectralFilter.get()==0:
        checkbutton.config(state=ACTIVE)
        button2['state'] = DISABLED
        entry2.delete(0)
        entry2.config(state=DISABLED)
        
##### Ausgewälte Filterung Ausführen

newFilter = False
         
def RunFilter():
    
    global FilterState ,HSSig, HSSigMean, HSSigMedian, HSSigSVN, newFilter, DrawState, newColor, MaskWidth, SegState, Index, dlg, label_im, signame, Derivate
    
    ##### Räumliche Filterung Auswahl Prüfen
    
    if SpacialFilter.get()==1:
        
        ##### Ausgewälte Filtergröße einlesen
            
        FilterSize = int(entry.get()) 
            
        ##### Ausgewählter räumliche Filter Prüfen
            
        if FilterChoice.get()==1:
                
            ##### Mittelwert Filter ausführen
                
            HSSigMean = ndimage.uniform_filter(HSSig, size=FilterSize)
            
            if Derivate.get()==1:
                HSSigMean = np.pad (HSSigMean, pad_width = ((0,0),(0,0),(1,1)), mode='symmetric')
                HSSigMean = np.diff (HSSigMean, n = 2, axis = 2)
                HSSigMean = ndimage.uniform_filter(HSSigMean, size = 7)
                HSSigMean = savgol_filter(HSSigMean, window_length= 9 , polyorder= 2, axis = 2, mode = 'mirror')
                
            FilterState = 'Mean Filter'
            label3['text'] = signame + '\n' 'Meaning Filtering of size ' + entry.get() + ' x ' + entry.get()
            newFilter = True
            
            if DrawState == True:
                button6['state'] = ACTIVE
       
        else:
                
            ##### Median Filter ausführen
                
            HSSigMedian = ndimage.median_filter(HSSig, size=FilterSize)
            
            if Derivate.get()==1:
                HSSigMedian = np.pad (HSSigMedian, pad_width = ((0,0),(0,0),(1,1)), mode='symmetric')
                HSSigMedian = np.diff (HSSigMedian, n = 2, axis = 2)
                HSSigMedian = ndimage.uniform_filter(HSSigMedian, size = 7)
                HSSigMedian = savgol_filter(HSSigMedian, window_length= 9 , polyorder= 2, axis = 2, mode = 'mirror')
            FilterState = 'Median Filter'
            label3['text'] = signame + '\n' 'Median Filtering of size ' + entry.get() + ' x ' + entry.get()
            newFilter = True
            
            if DrawState == True:
                button6['state'] = ACTIVE
                
    ##### Spektrale Filterung Auswahl Prüfen        
    
    if SpectralFilter.get()==1:
        
        MaskWidth = int(entry2.get())
        HSSigMeanSVN = ndimage.uniform_filter(HSSig, size=MaskWidth)
        mean = np.mean(HSSigMeanSVN, axis=2)
        std = np.std(HSSigMeanSVN, axis=2)
        std [std==0] = 1
        ##### Spektrale Filterung SVN ausführen
        
        HSSigSVN=(HSSigMeanSVN - mean[..., np.newaxis])/std[..., np.newaxis]
        
        if Derivate.get()==1:
           HSSigSVN = np.pad (HSSigSVN, pad_width = ((0,0),(0,0),(1,1)), mode='symmetric')
           HSSigSVN = np.diff (HSSigSVN, n = 2, axis = 2)
           HSSigSVN = ndimage.uniform_filter(HSSigSVN, size = 7)
           HSSigSVN = savgol_filter(HSSigSVN, window_length= 9 , polyorder= 2, axis = 2, mode = 'mirror')
           # HSSigSVN = ndimage.uniform_filter(HSSigSVN, size = 9)
           # HSSigSVN = ndimage.median_filter(HSSigSVN, size = (1,1,3))
        FilterState = 'SVN'
        label3['text'] = signame + '\n' 'SVN Filtering in '+ entry2.get() + ' x ' + entry2.get()+ ' points'
        newFilter = True
        
        if SegState == True and dlg.result == "Another data":
            
            HSSigSVN = np.multiply(HSSigSVN, Index[:,:,None])
            
        elif SegState == True and dlg.result == "Thyroid gland":
            
            HSSigSVN = np.multiply(HSSigSVN, label_im[:,:,None])
        
        if DrawState == True:
            button6['state'] = ACTIVE
    
    button3.config(state=ACTIVE)
    button2['state'] = DISABLED
    SpacialFilter.set(0)
    SpectralFilter.set(0)
    checkbutton.config(state=ACTIVE)
    checkbutton2.config(state=ACTIVE)
    FilterChoice.set(0)
    rb1.config(state = DISABLED)
    rb2.config(state = DISABLED)
    label2.config(state = DISABLED)
    entry.delete(0)
    entry.config(state=DISABLED)
    entry2.delete(0)
    entry2.config(state=DISABLED)
    button.config(state=DISABLED)
    AddClearState.set(0)
    ColorChanged == False
    newColor = False
    
##### Zeichnung Click Klass

class Click():
    def __init__(self, ax, func, button=1):
        self.ax=ax
        self.func=func
        self.button=button
        self.press=False
        self.move = False
        self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def onclick(self,event):
        if event.inaxes == self.ax:
            if event.button == self.button:
                self.func(event, self.ax)
    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False

##### Spektrum darstellen in ausgewählten Bildunkten mit Mausfunktion 

newColor = False
DrawState = False
Xcoords = []
Ycoords = []

def DrawSpecta ():
    
    global HSSig, HSSigMean, HSSigMedian, HSSigSVN, rgb_image, f, ax, ax2,GraphFrame, IMAGE, cursor, cid, K, FilterState, SpecColor,ColorChanged, SpecColorChange, Color, newColor, WLD
    
    if AddClearState.get() == 1:
        
        if newColor == False:
            
            SpecColor = askcolor(title = "Choose Spectra Color")
            Color = SpecColor[1]
            
            while SpecColor[1] == None:
                
                res = messagebox.showwarning('Drawing Color', 'please select one color')
                
                if res == None:
                    
                    res = messagebox.showwarning('Drawing Color', 'please select one color')
                    
                else :
                    
                    SpecColor = askcolor(title = "Choose Spectra Color")
                    Color = SpecColor[1]         

        def onclick(event, ax):
        
            global K, Color, SpecColorChange, ColorChanged, newColor, newFilter, DrawState, Ycoords, Xcoords, OverdrawingState, cid, WLD
            
            if OverdrawingState == True or newColor == False:
                
                Xcoords = []
                Ycoords = [] 
                newColor = True
                OverdrawingState = False
                
            x, y = int(event.xdata), int(event.ydata)
            Xcoords.append(x)
            Ycoords.append(y)
            
            if ColorChanged == True and SpecColorChange[1] != None: 
                Color = SpecColorChange[1]
                ColorChanged = False
                newFilter = True
                
            if event.inaxes and AddClearState.get() == 1:
        
                IMAGE = ax.scatter(x,y, s=30, c = Color, marker='x')
                
                if FilterState == 'Mean Filter':
                    
                    if newFilter == True:
                        ax2.plot(WLD, HSSigMean[y, x, :], color = Color, label=label3['text'])
                        ax2.legend()
                        newFilter = False
                        
                    elif newFilter == False:
                        ax2.plot(WLD, HSSigMean[y, x, :], color = Color)
                    
                elif FilterState == 'Median Filter':
                    
                    if newFilter == True:
                        ax2.plot(WLD,HSSigMedian[y, x, :], color = Color, label=label3['text'])
                        ax2.legend()
                        newFilter = False
                        
                    elif newFilter == False:
                        ax2.plot(WLD,HSSigMedian[y, x, :], color = Color)
                    
                elif FilterState == 'SVN':
                    
                    if newFilter == True:
                        ax2.plot(WLD,HSSigSVN[y, x, :], color = Color, label=label3['text'])
                        ax2.legend()
                        newFilter = False
                        
                    elif newFilter == False:
                        ax2.plot(WLD,HSSigSVN[y, x, :], color = Color)
                    
                elif FilterState == 'No Filter':
                        
                    if newFilter == True:
                        ax2.plot(WLD,HSSig[y, x, :], color = Color, label=label3['text'])
                        ax2.legend()
                        newFilter = False
                        
                    elif newFilter == False:
                        ax2.plot(WLD,HSSig[y, x, :], color = Color)
                
                IMAGE.figure.canvas.draw()
                DrawState = True
            
            button4.config(state=ACTIVE)
            button.config(state=DISABLED)
            button7.config(state=ACTIVE)
            button8.config(state=ACTIVE)
            
        button5.config(state=ACTIVE)
        click = Click(ax, onclick, button=1)
        # cid = f.canvas.mpl_connect('button_press_event', onclick(ax))
        cid = f.canvas.mpl_connect('button_press_event',
                                   lambda event: onclick(event, ax))

##### Grafik Aktualisieren

def ActualizeGraph ():
    
    global f, ax, ax2, GraphFrame, cursor, ColorChanged, FilterState, DrawState, newColor, Coord, SpecX, S, newFilter, signame
    
    res = messagebox.askquestion('clear Spectra', 'Do you really want to clear Spectra') 
    
    if res == 'yes' :

        ax.clear()
        ax2.clear()
        
        GraphFrame.place_forget()
        GraphFrame = tk.Frame(root, bg='gray')
        GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)
                    
        f = Figure()
        ax = f.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
        ax.imshow(rgb_image)
        ax2 = f.add_subplot(122, title = 'HS-Signal spectra', xlabel='Wave length')
        ax2.grid()
                
        canvas = FigureCanvasTkAgg(f, GraphFrame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas._tkcanvas.pack()
            
        cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
        NavigationToolbar2Tk.toolitems = (
                                      ('Home', 'Reset original view', 'home', 'home'),
                                      ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                                      ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                                      ('Save', 'Save the figure', 'filesave', 'save_figure'),
                                     )
        toolbar = NavigationToolbar2Tk(canvas, GraphFrame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)    
        
        AddClearState.set(0)
        
        button4['state'] = DISABLED
        button5['state'] = DISABLED
        ColorChanged = False
        button6['state'] = DISABLED
        button7.config(state=DISABLED)
        button8.config(state=DISABLED)
        button3.config(state=ACTIVE)
        button.config(state=ACTIVE)
        DrawState = False
        newColor = False
        newFilter = True
        FilterState = 'No Filter'
        label3['text'] = signame + '\n' 'Spectra without Filtering'
        Coord = []
        SpecX = []
        S = 0
##### Darstellungsfarbe ändern

ColorChanged = False

def ChangeColor():
    
    global ColorChanged, SpecColorChange, newColor, newFilter
    newColor = True
    ColorChanged = True
    SpecColorChange = askcolor(title = "Choose Spectra Color")

##### Überzeichnen und Vergleichen

OverdrawingState = False

def OverDrawing ():
    
    global FilterState, HSSig, HSSigMean, HSSigMedian, HSSigSVN, rgb_image, f, ax, ax2, IMAGE, Xcoords, Ycoords, newFilter, OverdrawingState, WLD, ChangeHSSig
    
    ColorOver = askcolor(title = "Choose Spectra Color")  
    Color = ColorOver[1]
            
    while ColorOver[1] == None:
                
        res = messagebox.showwarning('Drawing Color', 'please select one color')
                
        if res == None:
                    
            res = messagebox.showwarning('Drawing Color', 'please select one color')
                    
        else :
                    
            ColorOver = askcolor(title = "Choose Spectra Color")
            Color = ColorOver[1]
    
    if FilterState == 'Mean Filter':
        
        for i, j in zip (Xcoords, Ycoords):
            
            if newFilter == False and ChangeHSSig == False:
                ax2.plot(WLD,HSSigMean[j, i, :], color = Color)
            
            else:
                ax2.plot(WLD,HSSigMean[j, i, :], color = Color, label=label3['text'])
                ax2.legend()
                newFilter = False
                ChangeHSSig = False
                
                      
    elif FilterState == 'Median Filter':
        
        for i , j in zip (Xcoords, Ycoords):
            
            if newFilter == False and ChangeHSSig == False:
                ax2.plot(WLD,HSSigMedian[j, i, :], color = Color)
                
            else:
                ax2.plot(WLD,HSSigMedian[j, i, :], color = Color, label=label3['text'])
                ax2.legend()
                newFilter = False
                ChangeHSSig = False
                
    elif FilterState == 'SVN':
        
        for i, j in zip (Xcoords, Ycoords):
            
            if newFilter == False and ChangeHSSig == False:
                ax2.plot(WLD,HSSigSVN[j, i, :], color = Color)
            
            else:
                ax2.plot(WLD,HSSigSVN[j, i, :], color = Color, label=label3['text'])
                ax2.legend()
                newFilter = False
                ChangeHSSig = False
                    
    elif FilterState == 'No Filter':
        
        for i, j in zip (Xcoords, Ycoords):
            
            if newFilter == False and ChangeHSSig == False:
                ax2.plot(WLD,HSSig[j, i, :], color = Color)
            
            else:
                ax2.plot(WLD,HSSig[j, i, :], color = Color, label=label3['text'])
                ax2.legend()
                newFilter = False
                ChangeHSSig = False
                
    ax2.figure.canvas.draw()
    button6['state'] = DISABLED
    OverdrawingState = True
    newFilter = True
    
##### Erzeugung ziehbare Linien für Auswahl der Spektren
Coord = []

class draggable_lines:
    def __init__(self, ax, kind, XorY, Col):
        
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.o = kind
        self.XorY = XorY
        self.Col = Col

        if kind == "h":
            x = [-1, 1]
            y = [XorY, XorY]

        elif kind == "v":
            x = [XorY, XorY]
            y = [-10, 10]
        self.line = lines.Line2D(x, y, picker=5, color=Col)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            # print("line selected ", event.artist)
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)

    def followmouse(self, event):
        if self.o == "h":
            self.line.set_ydata([event.ydata, event.ydata])
        else:
            self.line.set_xdata([event.xdata, event.xdata])
        self.c.draw_idle()

    def releaseonclick(self, event):
        
        global Coord
        
        if self.o == "h":
            self.XorY = self.line.get_ydata()[0]
        else:
            self.XorY = self.line.get_xdata()[0]

        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)
        
        Coord.append(self.XorY)
              
##### Spektrum Auswählen

SpecX = []
S = 0

def SelectSpectra():
    
    global Tline, Coord, SpecX, Xf, S
    
    Tline = draggable_lines(ax2, "v", 750, 'r')
    
    if S > 0:
        
        Xf = int(round((Coord[len(Coord)-1])/5)*5)
        SpecX.append(Xf)
    
    S += 1
    AddClearState.set(0)
    button6['state'] = DISABLED
    button3['state'] = DISABLED
    
##### Spektrum Abbilden

B = 0
    
def BuildStectra():
    
    global Coord, SpecX, Xf,HSSig, HSSigMean, HSSigMedian, HSSigSVN, FilterState, B, S
    
    if B > 0 :
        plt.close(fig = 'all')
        XfSpec = int(round((SpecX[len(SpecX)-1])/5)*5)
        Xf = int(round((Coord[len(Coord)-1])/5)*5)
        
        if XfSpec == Xf:
            SpecX = SpecX
            print (SpecX)
        else:
            SpecX.append(Xf)
            print (SpecX)     
    else:
    
        Xf = int(round((Coord[len(Coord)-1])/5)*5)
        SpecX.append(Xf)
        print (SpecX)
    
    cmap = np.loadtxt('cmap_tivita.txt')
    T_cmap = ListedColormap(cmap)
    
    for i in SpecX:
        
        if FilterState == 'No Filter':
            fig = plt.figure()
            Img = fig.add_subplot(111, title= label3['text'] + ' in Wave length ' + str(i) )
            Img.imshow(HSSig[:, :, int((i-500)/5)], cmap= T_cmap)
            
            plt.show()
            
        elif FilterState == 'Mean Filter':
            fig = plt.figure()
            Img = fig.add_subplot(111, title= label3['text'] + ' in Wave length ' + str(i) )
            Img.imshow(HSSigMean[:, :, int((i-500)/5)], cmap= T_cmap)
            
            plt.show()
            
        elif FilterState == 'Median Filter':
            fig = plt.figure()
            Img = fig.add_subplot(111, title= label3['text'] + ' in Wave length ' + str(i) )
            Img.imshow(HSSigMedian[:, :, int((i-500)/5)], cmap= T_cmap)
            
            plt.show()
            
        elif FilterState == 'SVN':
            fig = plt.figure()
            Img = fig.add_subplot(111, title= label3['text'] + ' in Wave length ' + str(i) )
            Img.imshow(HSSigSVN[:, :, int((i-500)/5)], cmap= T_cmap)
            
            plt.show()
            
    B += 1
    S = 0
    
##### Starten der Klassifizierung
    
def StartClassification():
    
    global f, ax, ax2, GraphFrame, cursor, DrawState
        
    if DrawState == True:
        
        res = messagebox.askquestion('Saving spectral drawing', 'Do you really want to start classification without saving drawed spectra')
        
        if res == 'yes':
        
            ax.clear()
            ax2.clear()
            
            GraphFrame.place_forget()
            GraphFrame = tk.Frame(root, bg='gray')
            GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)
                        
            f = Figure()
            ax = f.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
            ax.imshow(rgb_image)
            ax2 = f.add_subplot(122, title = 'HS-Signal spectra', xlabel='Wave length')
            ax2.grid()
                    
            canvas = FigureCanvasTkAgg(f, GraphFrame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            canvas._tkcanvas.pack()
                
            cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
            NavigationToolbar2Tk.toolitems = (
                                      ('Home', 'Reset original view', 'home', 'home'),
                                      ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                                      ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                                      ('Save', 'Save the figure', 'filesave', 'save_figure'),
                                     )
            toolbar = NavigationToolbar2Tk(canvas, GraphFrame)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
            
            button.config(state=DISABLED)
            #button2.config(state=DISABLED)
            button3.config(state=DISABLED)
            button4.config(state=DISABLED)
            button5.config(state=DISABLED)
            button6.config(state=DISABLED)
            button7.config(state=DISABLED)
            button8.config(state=DISABLED)
            button9.config(state=DISABLED)
            #checkbutton.config(state=DISABLED)
            #checkbutton2.config(state=DISABLED)
            button11.config(state=ACTIVE)
            button16.config(state=ACTIVE)
    
    else:
        
        ax.clear()
        ax2.clear()
        
        GraphFrame.place_forget()
        GraphFrame = tk.Frame(root, bg='gray')
        GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)
                        
        f = Figure()
        ax = f.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
        ax.imshow(rgb_image)
        ax2 = f.add_subplot(122, title = 'HS-Signal spectra', xlabel='Wave length')
        ax2.grid()
                    
        canvas = FigureCanvasTkAgg(f, GraphFrame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas._tkcanvas.pack()
                
        cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
        NavigationToolbar2Tk.toolitems = (
                                      ('Home', 'Reset original view', 'home', 'home'),
                                      ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                                      ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                                      ('Save', 'Save the figure', 'filesave', 'save_figure'),
                                     )
        toolbar = NavigationToolbar2Tk(canvas, GraphFrame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
            
        button.config(state=DISABLED)
        #button2.config(state=DISABLED)
        button3.config(state=DISABLED)
        button4.config(state=DISABLED)
        button5.config(state=DISABLED)
        button6.config(state=DISABLED)
        button7.config(state=DISABLED)
        button8.config(state=DISABLED)
        button9.config(state=DISABLED)
        # checkbutton.config(state=DISABLED)
        # checkbutton2.config(state=DISABLED)
        
        button11.config(state=ACTIVE)
        button16.config(state=ACTIVE)
        
##### Klasse Inizialisieren

NewClass = False
Class = 0
ClassNames = []

def AddClass ():
    
    global ClassName, NewClass, Class
    ClassName = simpledialog.askstring("Class Name", "Please enter the Class name")
    while ClassName == '':
        
        res = messagebox.showwarning('Class Name', 'Please enter the Class name')
                
        if res == None:
                    
            res = messagebox.showwarning('Class Name', 'no class name has been entered')
                    
        else :
                    
            ClassName = simpledialog.askstring("Class Name", "Please enter the Class name")
        
    if ClassName != None:
        ClassNames.append(ClassName)
        button12.config(state=ACTIVE)
        Class += 1
        NewClass = True
    
    AddFeatur.set(0)
    
##### Klassen-Featurs hinzufühgen

Xtrain = []
Ytrain = []
ClColor = []

def AddClassFeaturs():
    
    global ClassName, NewClass, Xtrain, Ytrain, ClassColor, Class, cid, ClColor, WLD
    
    if AddFeatur.get() == 1:
        
        if NewClass == True:
            
            ClassColor = askcolor(title = "Choose Spectra Color")
            Color = ClassColor[1]
            
            while ClassColor[1] == None:
                
                res = messagebox.showwarning('Drawing Color', 'please select one color')
                
                if res == None:
                    
                    res = messagebox.showwarning('Drawing Color', 'please select one color')
                    
                else :
                    
                    ClassColor = askcolor(title = "Choose Spectra Color")
                    Color = ClassColor[1]
            
            ClColor.append(Color)
            
            def onclick(event, ax):
                
                global ClassName, NewClass, Xtrain, Ytrain, ClassColor, Class, cid
                
                x, y = int(event.xdata), int(event.ydata)
                button14.config(state=ACTIVE)
                
                if Class > 1:
                    
                    button13.config(state=ACTIVE)
                
                if event.inaxes and AddFeatur.get() == 1:
        
                    IMAGE = ax.scatter(x,y, s=30, c = ClassColor[1], marker='x')
                
                    if FilterState == 'Mean Filter':
                    
                        if NewClass == True:
                            ax2.plot(WLD,HSSigMean[y, x, :], color = ClassColor[1], label=ClassName)
                            ax2.legend()
                            Xtrain.append(HSSigMean[y, x, :])
                            Ytrain.append(Class)
                            NewClass = False
                            
                        elif NewClass == False:
                            ax2.plot(WLD,HSSigMean[y, x, :], color = ClassColor[1])
                            Xtrain.append(HSSigMean[y, x, :])
                            Ytrain.append(Class)
                            
                    elif FilterState == 'Median Filter':
                        
                        if NewClass == True:
                            ax2.plot(WLD,HSSigMedian[y, x, :], color = ClassColor[1], label=ClassName)
                            ax2.legend()
                            Xtrain.append(HSSigMedian[y, x, :])
                            Ytrain.append(Class)
                            NewClass = False
                            
                        elif NewClass == False:
                            ax2.plot(WLD,HSSigMedian[y, x, :], color = ClassColor[1])
                            Xtrain.append(HSSigMedian[y, x, :])
                            Ytrain.append(Class)
                            
                    elif FilterState == 'SVN':
                        
                        if NewClass == True:
                            ax2.plot(WLD,HSSigSVN[y, x, :], color = ClassColor[1], label=ClassName)
                            ax2.legend()
                            Xtrain.append(HSSigSVN[y, x, :])
                            Ytrain.append(Class)
                            NewClass = False
                            
                        elif NewClass == False:
                            ax2.plot(WLD,HSSigSVN[y, x, :], color = ClassColor[1])
                            Xtrain.append(HSSigSVN[y, x, :])
                            Ytrain.append(Class)
                    
                    elif FilterState == 'No Filter':
                        
                        if NewClass == True:
                            ax2.plot(WLD,HSSig[y, x, :], color = ClassColor[1], label=ClassName)
                            ax2.legend()
                            Xtrain.append(HSSig[y, x, :])
                            Ytrain.append(Class)
                            NewClass = False
                            
                        elif NewClass == False:
                            ax2.plot(WLD,HSSig[y, x, :], color = ClassColor[1])
                            Xtrain.append(HSSig[y, x, :])
                            Ytrain.append(Class)
                        
                    IMAGE.figure.canvas.draw()
            
            # AddFeatur.set(0)
            click = Click(ax, onclick, button=1)
            # cid = f.canvas.mpl_connect('button_press_event', onclick(ax))
            cid = f.canvas.mpl_connect('button_press_event',
                                       lambda event: onclick(event, ax))
##### Ausgewählte Klassen bestätigen
        
def ConfirmSelectedClasses ():
    
    global Xtest, NRows, NCols, NBands, Xtrain, Ytrain
    
    selectClassifier.configure(state=ACTIVE)
    button10.config(state=ACTIVE)
    
    button9.config(state=DISABLED)
    button11.config(state=DISABLED)
    AddFeatur.set(0)
    button12.config(state=DISABLED)
    button13.config(state=DISABLED)
    
    if FilterState == 'Mean Filter':
        Xtest = HSSigMean.reshape (NRows*NCols, NBands)
    elif FilterState == 'Median Filter':
        Xtest = HSSigMedian.reshape (NRows*NCols, NBands)
    elif FilterState == 'SVN':
        Xtest = HSSigSVN.reshape (NRows*NCols, NBands)
    elif FilterState == 'No Filter':
        Xtest = HSSig.reshape (NRows*NCols, NBands)
        
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
        
##### Classen Löschen und Classenvariabeln Zurücksetzen

def ClearClasses (): 
    
    global Xtrain, Ytrain, NewClass, ClassNames, ClClear, f, ax, ax2, GraphFrame, cursor, Class, ClColor
    
    res = messagebox.askquestion('clear Classes and Features', 'Do you really want to clear Classes and Features') 
    
    if res == 'yes' :
    
        button11.config(state=ACTIVE)
        
        selectClassifier.configure(state=DISABLED)
        button9.config(state=ACTIVE)
        button10.config(state=DISABLED)
        AddFeatur.set(0)
        button12.config(state=DISABLED)
        button13.config(state=DISABLED)
        button14.config(state=DISABLED)
        button15.config(state=DISABLED)
        
        Xtrain = []
        Ytrain = []
        ClassNames = []
        ClClear = []
        ClColor = []
        NewClass = False
        Class = 0
        
        ax.clear()
        ax2.clear()
        
        GraphFrame.place_forget()
        GraphFrame = tk.Frame(root, bg='gray')
        GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)
        
        f = Figure()
        ax = f.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
        ax.imshow(rgb_image)
        ax2 = f.add_subplot(122, title = 'HS-Signal spectra', xlabel='Wave length')
        ax2.grid()
                
        canvas = FigureCanvasTkAgg(f, GraphFrame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas._tkcanvas.pack()
            
        cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
        NavigationToolbar2Tk.toolitems = (
                                      ('Home', 'Reset original view', 'home', 'home'),
                                      ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                                      ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                                      ('Save', 'Save the figure', 'filesave', 'save_figure'),
                                     )
        toolbar = NavigationToolbar2Tk(canvas, GraphFrame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
##### Spektrale Klassifizierunsbereich auswählen

def SelectSpectralRange ():
    
    global SpecI, SpecF, SpecLim
    
    SpecI = draggable_lines(ax2, "v", 500, 'g')
    SpecF = draggable_lines(ax2, "v", 1000, 'r')
    selectClassifier.configure(state=ACTIVE)
    button15.config(state=ACTIVE)
    button10.config(state=DISABLED)
##### Klassifizierung ausführen

Means = []
Mcoords = []

def RunClassification():
    
    global SpecI, SpecFn, Xtrain, Ytrain, Xtest, var, Xpi, Xpf, Means, Mcoords, rgb_image, Class, ClColor
    
    #int(round((SpecX[len(SpecX)-1])/5)*5)
    
    Xpi = int (round((min((SpecI.line.get_xdata()[0]), (SpecF.line.get_xdata()[0])))/5)*5)
    Xpf = int (round((max((SpecI.line.get_xdata()[0]), (SpecF.line.get_xdata()[0])))/5)*5)
    
    Xpi = int((Xpi-500)/5)
    Xpf = int((Xpf-500)/5)
    
    XtrainEff = Xtrain[:,Xpi:Xpf]
    XtrainEff[XtrainEff == 0] = 0.2
    XtestEff  = Xtest[: ,Xpi:Xpf]
    XtestEff[XtestEff == 0] = 0.2
    
    if var.get() == 'SAM':
        
        XSAMtest = XtestEff
        XSAMtrain = XtrainEff
        
        for i in range (1, max(Ytrain)+1):
            
            Mcoords.append(np.where(Ytrain == i))
            
        for i in range (len(Mcoords)):
            
            Min = int (np.min(np.array(Mcoords[i])))
            Max = int (np.max(np.array(Mcoords[i])))
            mean = XSAMtrain[Min:Max, :].mean(axis = 0)
            Means.append(mean)
        Means = np.array(Means)
        plt.figure()
        
        for i in range(Means.shape[0]):
            plt.plot(Means[i, :], color = ClColor[i])
        plt.grid()
        plt.show()
        
        XSAMtest = XSAMtest.reshape (NRows, NCols, Xpf-Xpi)
        angles = spectral_angles(XSAMtest, Means)
        ClassOut = np.argmin(angles, 2)
        
        Means = []
        Mcoords = []
        
    elif var.get() == 'PCA':
        
        XtrainPCA = XtrainEff
        XtestPCA = XtestEff
        
        pca = PCA(svd_solver='full')
        XtrainPCA = pca.fit_transform(XtrainPCA)
        XtestPCA = pca.transform(XtestPCA)
        
        classifier = RandomForestClassifier(max_depth=2, random_state=0)
        classifier.fit(XtrainPCA, Ytrain)
        
        y_pred = classifier.predict(XtestPCA)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    elif var.get() == 'KPCA':
        
        XtrainKPCA = XtrainEff
        XtestKPCA = XtestEff
        
        kpca = KernelPCA(n_components=Xpf-Xpi, kernel='cosine')
        XtrainKPCA = kpca.fit_transform(XtrainKPCA)
        XtestKPCA = kpca.transform(XtestKPCA)
        
        classifier = RandomForestClassifier(max_depth=2, random_state=0)
        classifier.fit(XtrainKPCA, Ytrain)
        
        y_pred = classifier.predict(XtestKPCA)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    elif var.get() == 'LDA':
        
        XtrainLDA = XtrainEff
        XtestLDA = XtestEff
        
        lda = LDA(n_components=Class-1)
        XtrainLDA = lda.fit_transform(XtrainLDA, Ytrain)
        XtestLDA = lda.transform(XtestLDA)
        
        classifier = RandomForestClassifier(max_depth=2, random_state=0)
        classifier.fit(XtrainLDA, Ytrain)

        y_pred = classifier.predict(XtestLDA)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    elif var.get() == 'GPC':
        
        XtrainGPC = XtrainEff
        XtestGPC = XtestEff
        
        kernel = 1.0 * RBF(1.0)
        gpc = GaussianProcessClassifier(kernel=kernel, max_iter_predict=500 ,random_state=0).fit(XtrainGPC, Ytrain)
        
        y_pred = gpc.predict(XtestGPC)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    elif var.get() == 'MLP':
        
        XtrainMLP = XtrainEff
        XtestMLP = XtestEff
        
        clf = MLPClassifier(random_state=1, max_iter=200000).fit(XtrainMLP, Ytrain)
        y_pred = clf.predict(XtestMLP)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    elif var.get() == 'KNN':
        
        XtrainKNN = XtrainEff
        XtestKNN = XtestEff
        
        neigh = KNeighborsClassifier(n_neighbors=42, weights='distance').fit(XtrainKNN, Ytrain)    
        y_pred = neigh.predict(XtestKNN)
        ClassOut = y_pred.reshape(NRows, NCols)
    
    elif var.get() == 'SVM':
        
        XtrainSVM = XtrainEff
        XtestSVM = XtestEff
        
        clf = SVC().fit(XtrainSVM, Ytrain)    
        y_pred = clf.predict(XtestSVM)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    elif var.get() == 'SOM':
        
        XtrainSOM = XtrainEff
        XtestSOM = XtestEff
        
        som = susi.SOMRegressor(
        n_rows=35,
        n_columns=35,
        n_iter_unsupervised=2500,
        n_iter_supervised=2500,
        neighborhood_mode_unsupervised="linear",
        neighborhood_mode_supervised="linear",
        learn_mode_unsupervised="min",
        learn_mode_supervised="min",
        learning_rate_start=0.5,
        learning_rate_end=0.05,
        random_state=None,
        n_jobs=1)
        
        som.fit(XtrainSOM, Ytrain)    
        y_pred = som.predict(XtestSOM)
        ClassOut = y_pred.reshape(NRows, NCols)
        
    T_cmap = LinearSegmentedColormap.from_list('ClassColors', ClColor, N=len(ClColor))
        
    plt.figure()
        
    plt.subplot(121)
    plt.imshow(rgb_image)
    plt.xticks([]), plt.yticks([])
        
    plt.subplot(122)
    plt.imshow(ClassOut, cmap =T_cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
##### Klassen und Features copieren und auf einem neuen HS-Signal anwenden

def CopyClassAndFeatures():
    
    res = messagebox.askquestion('Copy Classes and Features', 'for the comparison of the current classification '
                                 'results with a neu classification using an another HS-Signal, all class names and'
                                 ' training dataset (spectral features in the same image points and with the '
                                 'same colors) will be copied and applied to a new selected HS signal.' 
                                 + '\n' '\n'+ 'if you are ok with that press  "Yes" .'+ '\n' '\n'
                                 + '!!Warning!! : all the current spectra will be replaced with the neu spectra of '
                                 'selected HS-Signal.' + '\n' + 'Please save the Figure of the current training dataset'
                                 ' if it is necessery.'+ '\n''\n' + 'press  "no"  if you want to save the figure') 
                                 
    
    if res == 'yes' : 
        
        print('OK')
    
##### Klassifizierung Verlassen

def QuitClassification():
    
    global var, Xtrain, Ytrain, NewClass, ClassNames, ClClear, f, ax, ax2, GraphFrame, cursor, Class, ClColor
    
    res = messagebox.askquestion('Quit Classification', 'Do you really want to quit the classification without saving the results') 
    
    if res == 'yes' :
    
        button.config(state=ACTIVE)
        checkbutton.config(state=ACTIVE)
        checkbutton2.config(state=ACTIVE)
        button3.config(state=ACTIVE)
        button9.config(state=ACTIVE)
        button13.config(state=DISABLED)
        button14.config(state=DISABLED)
        button15.config(state=DISABLED)
        button16.config(state=DISABLED)
        var.set('Select Classifier')
        selectClassifier.configure(state=DISABLED)
        
        Xtrain = []
        Ytrain = []
        ClassNames = []
        ClClear = []
        ClColor = []
        NewClass = False
        Class = 0
        
        ax.clear()
        ax2.clear()
        
        GraphFrame.place_forget()
        GraphFrame = tk.Frame(root, bg='gray')
        GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)
                    
        f = Figure()
        ax = f.add_subplot(121, xticks = [], yticks = [], title = 'RGB Image')
        ax.imshow(rgb_image)
        ax2 = f.add_subplot(122, title = 'HS-Signal spectra', xlabel='Wave length')
        ax2.grid()
                
        canvas = FigureCanvasTkAgg(f, GraphFrame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas._tkcanvas.pack()
            
        cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
        NavigationToolbar2Tk.toolitems = (
                                      ('Home', 'Reset original view', 'home', 'home'),
                                      ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                                      ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                                      ('Save', 'Save the figure', 'filesave', 'save_figure'),
                                     )
        toolbar = NavigationToolbar2Tk(canvas, GraphFrame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

##### Tkinter Gui: Frames und Button

root = tk.Tk()
root.wm_title("HS-Data Analysis")

canvas = tk.Canvas(root, height=700, width=1500)
canvas.pack()

CommandFrame = tk.Frame(root)
CommandFrame.place(relx=0.01, rely= 0.01,relwidth=0.12, relheight=0.98)

GraphFrame = tk.Frame(root, bg='gray')
GraphFrame.place(relx=0.14, rely= 0.01, relwidth=0.855, relheight=0.98)

label = tk.Label(CommandFrame, text="Parameter", font = 50)
label.place(relx = 0.01, rely = 0, relwidth = 0.98, relheight = 0.03)

button = tk.Button(CommandFrame, text ="Load Cube Data", command= LoadCubeData)
button.place(relx = 0.01, rely = 0.03, relwidth = 0.98, relheight = 0.035)

Signals = ('Intensity', 'Absorption', 'Residium from Intensity', 'Residium from Absorption')
varSig = tk.StringVar(CommandFrame)
varSig.set('Select HS-Signal')
selectHSSignal = tk.OptionMenu(CommandFrame, varSig, *Signals, command = SelectHsSignal)
selectHSSignal.place(relx = 0.01, rely = 0.11, relwidth = 0.98, relheight = 0.035)
selectHSSignal.configure(state=DISABLED)

buttonS = tk.Button(CommandFrame, text ="Segmentation", state=DISABLED, command = Segmentation)
buttonS.place(relx = 0.01, rely = 0.07, relwidth = 0.98, relheight = 0.035)

label2 = tk.Label(CommandFrame, text= "Select a filtering method")
label2.place(relx = 0.01, rely = 0.145, relwidth = 0.98, relheight = 0.03)

SpacialFilter = IntVar()
checkbutton = tk.Checkbutton(CommandFrame, text = "Spacial filter", variable= SpacialFilter, state=DISABLED, command = CheckSpacial)
checkbutton.place(relx = 0.01, rely = 0.175,  relheight = 0.03)

FilterChoice = IntVar()
rb1 =  tk.Radiobutton(CommandFrame, text= "Average Filter",variable = FilterChoice, value=1, state=DISABLED, command = ShooseFilter)
rb1.place(relx = 0.01, rely = 0.205, relheight = 0.03)
rb2 =  tk.Radiobutton(CommandFrame, text= "Median filter", variable = FilterChoice, value=2, state=DISABLED, command = ShooseFilter)
rb2.place(relx = 0.01, rely = 0.235, relheight = 0.03)

label2 = tk.Label(CommandFrame, text= "Filter size", state=DISABLED )
label2.place(relx = 0.01, rely = 0.265, relheight = 0.03)

sv = StringVar()
sv.trace("w", lambda name, index, mode, sv=sv: callback(sv))
entry = tk.Entry(CommandFrame, textvariable = sv, state=DISABLED)
entry.place(relx = 0.7, rely = 0.265, relheight = 0.03)

SpectralFilter = IntVar()
checkbutton2 = tk.Checkbutton(CommandFrame, text = "Spectral filter SVN", variable= SpectralFilter, state=DISABLED, command = CheckSpectral)
checkbutton2.place(relx = 0.01, rely = 0.30, relheight = 0.03)

label3 = tk.Label(CommandFrame, text= "Width of Mask", state=DISABLED )
label3.place(relx = 0.01, rely = 0.33, relheight = 0.03)

nbp = StringVar()
nbp.trace("w", lambda name, index, mode, nbp=nbp: callback2(nbp))
entry2 = tk.Entry(CommandFrame, textvariable = nbp, state=DISABLED)
entry2.place(relx = 0.7, rely = 0.33, relheight = 0.03)

Derivate = IntVar()
checkbutton3 = tk.Checkbutton(CommandFrame, text = "Second Derivative", variable= Derivate, state=ACTIVE)
checkbutton3.place(relx = 0.01, rely = 0.36, relheight = 0.025)

button2 = tk.Button(CommandFrame, text ="Run Filtering", state=DISABLED, command = RunFilter)
button2.place(relx = 0.01, rely = 0.385, relwidth = 0.98, relheight = 0.035)

label3 = tk.Label(CommandFrame, text="Filtering State", font = ('Helvetica', '7'), state=DISABLED)
label3.place(relx = 0.01, rely = 0.42, relwidth = 0.98, relheight = 0.03)

AddClearState = tk.IntVar()
button3 = tk.Radiobutton(CommandFrame, text ="Add Spectra", variable = AddClearState, value=1, indicatoron=0, font = ('Helvetica', '8'), state=DISABLED, command = DrawSpecta)
button3.place(relx = 0.01, rely = 0.455, relwidth = 0.48, relheight = 0.04)

button4 = tk.Button(CommandFrame, text ="Clear Spectra", font = ('Helvetica', '8'), state=DISABLED, command = ActualizeGraph)
button4.place(relx = 0.51, rely = 0.455, relwidth = 0.48, relheight = 0.04)

button5 = tk.Button(CommandFrame, text ="Change drawing color", font = ('Helvetica', '8'), state=DISABLED, command = ChangeColor)
button5.place(relx = 0.01, rely = 0.50, relwidth = 0.98, relheight = 0.04)

button6 = tk.Button(CommandFrame, text ="Draw on the same points", font = ('Helvetica', '8'), state=DISABLED, command = OverDrawing)
button6.place(relx = 0.01, rely = 0.55, relwidth = 0.98, relheight = 0.04)

button7 = tk.Button(CommandFrame, text ="Select Spectra", font = ('Helvetica', '8'), state=DISABLED, command = SelectSpectra)
button7.place(relx = 0.01, rely = 0.60, relwidth = 0.75, relheight = 0.04)

button8 = tk.Button(CommandFrame, text ="Build", font = ('Helvetica', '8'), state=DISABLED, command = BuildStectra)
button8.place(relx = 0.78, rely = 0.60, relwidth = 0.2, relheight = 0.04)

label4 = tk.Label(CommandFrame, text="Classification")
label4.place(relx = 0.01, rely = 0.645, relwidth = 0.98, relheight = 0.02)

button9 = tk.Button(CommandFrame, text ="Start Classification", font = ('Helvetica', '8'), state=DISABLED, command = StartClassification)
button9.place(relx = 0.01, rely = 0.67, relwidth = 0.98, relheight = 0.04)

button11 = tk.Button(CommandFrame, text ="Add Class", font = ('Helvetica', '8'), state=DISABLED, command = AddClass)
button11.place(relx = 0.01, rely = 0.72, relwidth = 0.48, relheight = 0.04)

AddFeatur = IntVar()
button12 = tk.Radiobutton(CommandFrame, text ="Add Featur", variable = AddFeatur, value=1, indicatoron=0, font = ('Helvetica', '8'), state=DISABLED, command = AddClassFeaturs)
button12.place(relx = 0.51, rely = 0.72, relwidth = 0.48, relheight = 0.04)

button13 = tk.Button(CommandFrame, text ="Confirm", font = ('Helvetica', '8'), state=DISABLED, command = ConfirmSelectedClasses)
button13.place(relx = 0.01, rely = 0.77, relwidth = 0.48, relheight = 0.04)

button14 = tk.Button(CommandFrame, text ="Clear", font = ('Helvetica', '8'), state=DISABLED, command = ClearClasses)
button14.place(relx = 0.51, rely = 0.77, relwidth = 0.48, relheight = 0.04)

button10 = tk.Button(CommandFrame, text ="Select spectral range", font = ('Helvetica', '8'), state=DISABLED, command = SelectSpectralRange)
button10.place(relx = 0.01, rely = 0.82, relwidth = 0.98, relheight = 0.04)

choices = ('SAM', 'PCA', 'KPCA', 'LDA', 'GPC', 'MLP', 'KNN', 'SVM', 'SOM')
var = tk.StringVar(CommandFrame)
var.set('Select Classifier')
selectClassifier = tk.OptionMenu(CommandFrame, var, *choices)
selectClassifier.place(relx = 0.01, rely = 0.87, relwidth = 0.98, relheight = 0.04)
selectClassifier.configure(state=DISABLED)
selectClassifier.config(font=tkFont.Font(family='Helvetica', size=8))

button15 = tk.Button(CommandFrame, text ="Run" +"\n" +"Classification", font = ('Helvetica', '7'), state=DISABLED, command = RunClassification)
button15.place(relx = 0.51, rely = 0.915, relwidth = 0.48, relheight = 0.045)

button17 = tk.Button(CommandFrame, text ="Duplicat" + "\n" +"Classification", font = ('Helvetica', '7'), state=ACTIVE, command = CopyClassAndFeatures)
button17.place(relx = 0.01, rely = 0.915, relwidth = 0.48, relheight = 0.045)

button16 = tk.Button(CommandFrame, text ="Quit Classification", font = ('Helvetica', '8'), state=DISABLED, command = QuitClassification)
button16.place(relx = 0.01, rely = 0.97, relwidth = 0.98, relheight = 0.02999)

root.mainloop()
