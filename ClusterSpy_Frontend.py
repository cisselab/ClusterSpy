from tkinter import *
from tkinter import Label,filedialog,messagebox
from tkinter import ttk
from PIL import ImageTk,Image
from ClusterSpy_Backend import *
from ClusterSpy_Backend import AveImPixel,FalseImage,PseudoCluster,PlotCenters,BestZ,Intensity,ExportData,ExporttoCSV,MakeMovie
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import customtkinter as ctk
import matplotlib.pyplot as plt
import os
import csv
import cv2
import numpy as np
import statistics as stat
import tkinter as tk
import tifffile as tif

#Global Parameter handlers
global ClusterSpyDirectory
ClusterSpyDirectory = os.getcwd() #Program's current directory  
global UserConfigDirectory
UserConfigDirectory = "AdvSetConfig"
global BrightnessThresh
global LocalContrastThresh
global KMeansMinClusterThresh
global ZLayerHeightInput
global OutputFileRequest
global DisplayResizeX
global DisplayResizeY
DisplayResizeX = 450 
DisplayResizeY = 300 
GraphResizeX = 450
GraphResizeY = 300


#Parameters for Dual Channel Mode
global Dual
Dual = False
global BrightnessThresh2
global LocalContrastThresh2
global KMeansMinClusterThresh2
global ZLayerHeightInput2
global OutputFileRequest2

#Front End Function for running ClusterSpy

#Locate and Store Max Intensity Folder in MaxIntensityDirectory

def PassTifFile():
    global HyperStackFile
    global HyperStackLabel
    global HyperStack
    HyperStackFile = filedialog.askopenfilename()
    HyperStackLabel.destroy()
    HyperStackLabel = ctk.CTkLabel(EntryFrame, text = HyperStackFile.split("/")[-1])
    HyperStackLabel.grid(row = 0, column = 0, columnspan = 2)
    HyperStack = tif.imread(HyperStackFile)

def PassTifFile2():
    global HyperStackFile2
    global HyperStackLabel2
    global HyperStack2
    HyperStackFile2 = filedialog.askopenfilename()
    HyperStackLabel2.destroy()
    HyperStackLabel2 = ctk.CTkLabel(EntryFrame2, text = HyperStackFile2.split("/")[-1])
    HyperStackLabel2.grid(row = 0, column = 0, columnspan = 2)
    HyperStack2 = tif.imread(HyperStackFile2)
    
#Function Responsible for identifying bright pixel threshold
def lockinBright(Bright): 
    try:
        global Brightness
        Brightness = float(Bright)
        if Brightness<1:
            messagebox.showerror("Brightness Threshold Error", "Brightness Threshold should be greater than average.")
            return 0
        BrightnessLabel = ctk.CTkLabel(EntryFrame, text = str(Brightness))
        BrightnessLabel.grid(row = 5, column = 2)
    except:
        messagebox.showerror("Brightness Threshold Error"
                             , "Please define your brightness threshold as a decimal")
        return 0
        
def lockinBright2(Bright): 
    try:
        global Brightness2
        global BrightnessLabel2
        Brightness2 = float(Bright)
        if Brightness2<1:
            messagebox.showerror("Brightness Threshold Error", "Brightness Threshold should be greater than average.")
            return 0
        BrightnessLabel2 = ctk.CTkLabel(EntryFrame2, text = str(Brightness2))
        BrightnessLabel2.grid(row = 5, column = 2)
    except:
        messagebox.showerror("Brightness Threshold Error"
                             , "Please define your brightness threshold as a decimal")
        return 0

#Function responsible for identifying contrast threshold
def lockinContrast(Contrast): 
    try:
        global LocalContrastvar
        LocalContrastvar = float(Contrast)
        LocalContrastLabel = ctk.CTkLabel(EntryFrame, text = str(LocalContrastvar))
        LocalContrastLabel.grid(row =6, column = 2)
    except:
        messagebox.showerror("Pseudo-Cluster Threshold Error", "Please only put in Decimals")
        return 0
        
def lockinContrast2(Contrast): 
    try:
        global LocalContrastvar2
        global LocalContrastLabel2
        LocalContrastvar2 = float(Contrast)
        LocalContrastLabel2 = ctk.CTkLabel(EntryFrame2, text = str(LocalContrastvar2))
        LocalContrastLabel2.grid(row =6, column = 2)
    except:
        messagebox.showerror("Pseudo-Cluster Threshold Error", "Please only put in Decimals")
        return 0
    
#Function responcible for initializing K means parameters
def lockinKMeans(KMeans): 
    if int(KMeans.split("-")[0])>= int(KMeans.split("-")[1]):
        messagebox.showerror("K-Means Initialization Warning","The Max K-Means value must be bigger than the Min K-Means Value")
        return 0
    if int(KMeans.split("-")[0]) == 0:
        messagebox.showerror("K-Means Initialization Warning", "The Minimum K-Means Cannot be 0")
        return 0
    try:
        global KMeansMinMax
        KMeansMinMax = KMeans
        KMeansMinLabel = ctk.CTkLabel(EntryFrame, text = str(KMeans))
        KMeansMinLabel.grid(row = 7, column = 2)
    except:
        messagebox.showerror("K-Means Initialization Warning", "Integer range in the form: Min--Max")
        return 0
    
def lockinKMeans2(KMeans): 
    if int(KMeans.split("-")[0])>= int(KMeans.split("-")[1]):
        messagebox.showerror("K-Means Initialization Warning","The Max K-Means value must be bigger than the Min K-Means Value")
        return 0
    if int(KMeans.split("-")[0]) == 0:
        messagebox.showerror("K-Means Initialization Warning", "The Minimum K-Means Cannot be 0")
        return 0
    try:
        global KMeansMinMax2
        global KMeansMinLabel2
        KMeansMinMax2 = KMeans
        KMeansMinLabel2 = ctk.CTkLabel(EntryFrame2, text = str(KMeans))
        KMeansMinLabel2.grid(row = 7, column = 2)
    except:
        messagebox.showerror("K-Means Initialization Warning", "Integer range in the form: Min--Max")
        return 0
    
#Function Responsible for making Csv file for data output
def lockinFile():
    try:
        global writer
        global ClusterInfoFile
        global CSVFileName
        CSVFileName = str(ImageFolderName + ".csv")
        ClusterInfoFile = open(CSVFileName,"w")
        writer = csv.writer(ClusterInfoFile)
        writer.writerow(['TimeStamp','Cluster Name','X','Y',"Z","ClusterSize", "Max", "Average", "LocalBack"
                         ,"Background Ratio: " + str(Brightness) ,"Contrast Ratio: " + str(LocalContrastvar) 
                         ,"KMeans Range: " + str(KMeansMinMax)])
        OutputFileLabel = ctk.CTkLabel(EntryFrame, text = CSVFileName)
        OutputFileLabel.grid(row =10, column = 2)
    except:
        messagebox.showerror("File Name Warning", "Please enter a valid file name")    
        
def lockinFile2():
    try:
        global writer2
        global ClusterInfoFile2
        global OutputFileLabel2
        global CSVFileName2
        CSVFileName2 = str(ImageFolderName2 + ".csv")
        ClusterInfoFile2 = open(CSVFileName2,"w")
        writer2 = csv.writer(ClusterInfoFile2)
        writer2.writerow(['TimeStamp','Cluster Name','X','Y',"Z","ClusterSize", "Max", "Average", "LocalBack"
                         ,"Background Ratio: " + str(Brightness2) ,"Contrast Ratio: " + str(LocalContrastvar2) 
                         ,"KMeans Range: " + str(KMeansMinMax2)])
        OutputFileLabel2 = ctk.CTkLabel(EntryFrame2, text = CSVFileName2)
        OutputFileLabel2.grid(row =10, column = 2)
    except:
        messagebox.showerror("File Name Warning", "Please enter a valid file name")    
        
#Function Responsible for making Image Folder Directory
def CreateSaveDirectory(File): 
    try:
        global SaveDirectory
        global ImageFolderName
        SaveDirectory = str(filedialog.askdirectory())
        os.chdir(SaveDirectory)
        ImageFolderName = str(File)
        os.makedirs(ImageFolderName)
        lockinFile()
        ImageFolderLabel = ctk.CTkLabel(EntryFrame,text = ImageFolderName)
        ImageFolderLabel.grid(row = 11, column = 2)
        os.chdir(ClusterSpyDirectory)
    except:
        messagebox.showerror("Image Folder Warning","Please Check that the name is valid and not in use")
        
def CreateSaveDirectory2(File2): 
    try:
        global SaveDirectory2
        global ImageFolderName2
        global ImageFolderLabel2
        SaveDirectory2 = str(filedialog.askdirectory())
        os.chdir(SaveDirectory2)
        ImageFolderName2 = str(File2)
        os.makedirs(ImageFolderName2)
        lockinFile2()
        ImageFolderLabel2 = ctk.CTkLabel(EntryFrame2,text = ImageFolderName2)
        ImageFolderLabel2.grid(row = 11, column = 2)
        os.chdir(ClusterSpyDirectory)
    except:
        messagebox.showerror("Image Folder Warning", "Please Check that the name is valid and not in use")
        
#Master lock parameters button
def LockInParameters(Brightness,LocalContrast,KMeans): 
    lockinBright(Brightness)
    lockinContrast(LocalContrast)
    lockinKMeans(KMeans)
    SaveDirectoryRequest = ctk.CTkButton(EntryFrame, text = "Pick a Save directory",image = save_directory_image, command = lambda: CreateSaveDirectory(OutputFileRequest.get()))
    SaveDirectoryRequest.grid(row = 11, column =0, columnspan = 2,padx = 5, pady=5)
    
def LockInParameters2(Brightness,LocalContrast,KMeans): 
    global SaveDirectoryRequestOn2
    lockinBright2(Brightness)
    lockinContrast2(LocalContrast)
    lockinKMeans2(KMeans)
    SaveDirectoryRequestOn2 = ctk.CTkButton(EntryFrame2, text = "Pick a Save directory",image = save_directory_image, command = lambda: CreateSaveDirectory2(OutputFileRequest2.get()))
    SaveDirectoryRequestOn2.grid(row = 11, column =0, columnspan = 2,padx = 5, pady=5)
    LockParameterLabel = True
    
#Handles the execution of back end call
def MasterButton():
    Ready = messagebox.askquestion("Before you run", "Have you checked that all variables were inputted?")
    if Ready == "yes":
        ClusterDetectionMaster(HyperStack
                               ,Brightness
                               ,LocalContrastvar
                               ,KMeansMinMax
                               ,SaveDirectory
                               ,ImageFolderName
                               ,writer
                               ,ClusterInfoFile
                               ,"1")
        if Dual:
            ClusterDetectionMaster(HyperStack2
                                   ,Brightness2
                                   ,LocalContrastvar2
                                   ,KMeansMinMax2
                                   ,SaveDirectory2
                                   ,ImageFolderName2
                                   ,writer2
                                   ,ClusterInfoFile2
                                   ,"2")
            
        messagebox.showinfo("Runtime complete", "ClusterSpy finished the task!")
        
        global DisplayProcessedImageButton
        DisplayProcessedImageButton.destroy()
        DisplayProcessedImageButton = ctk.CTkButton(EntryFrame, text = "Display", command = DisplayImageMaster)
        DisplayProcessedImageButton.grid(row = 2, column = 2)
    else:
        return 0
    
# #Initializes the Backend of the program to run. Also in charge of intensity calculation
def ClusterDetectionMaster(HStack,Bright,ContrastRatio,K,SaveD,ImgFolder,Pencil,Sheet,ProgressTag):
    global CenterMaster #Stores centers per frame Gets update in ClusterCellIndividual()
    #ProgressBar for Cluster Detection
    ClusterDetectionValue = 0
    ClusterProgress = ctk.CTkProgressBar(root, width = 300)
    ClusterProgress.grid(row = 1, column = 1)
    ClusterProgress.set(ClusterDetectionValue)
    ClusterProgressLabel = ctk.CTkLabel(root, text = "Finding Clusters Channel " + ProgressTag)
    ClusterProgressLabel.grid(row = 0, column = 1)
    
    #ProgressBar for Image Generation
    ImageGenerationValue = 0
    ImageGenerationProgress = ctk.CTkProgressBar(root, width = 300)
    ImageGenerationProgress.grid(row = 1, column = 2)
    ImageGenerationProgress.set(ImageGenerationValue)
    ImageGenerationProgressLabel = ctk.CTkLabel(root, text = "Generating Output")
    ImageGenerationProgressLabel.grid(row = 0, column = 2)
    
    TimeStamp = 0 #Counts the Time component
    CenterMaster = {} 
    ImageCount = HStack.shape[0]
    for Stack in HStack: #Eithers calls stack in case of Hyperstack or calls frame in case of MIP
        try:
            ClusterPositions= ClusterDetectionIndividual(Stack
                                                         ,Bright
                                                         ,ContrastRatio
                                                         ,K)
            CenterMaster[TimeStamp] = ClusterPositions    # {TimeStamp} = [[[XYZ],size],[[XYZ],size]]
        except ValueError:
            pass
        TimeStamp +=1
        ClusterDetectionValue += 1/ImageCount
        ClusterProgress.set(ClusterDetectionValue)
        root.update()
        
    TrackedClusterMasterList = TrackClusterBetweenFrames(CenterMaster,ClusterDiameter)
    LabelStep = len(TrackedClusterMasterList)
    
    for i in range(len(TrackedClusterMasterList)):
        FrameAndCenters = TrackedClusterMasterList[i]
        FrameName = FrameAndCenters[0]
        #Checks if HStack is MIP or hyperstack
        if len(HStack[i].shape) == 3:
            MIP = np.max(HStack[i], axis = 0) #Generates Max Intensity Projection from Hyperstack
        elif len(HStack[i].shape) == 2:
            MIP = HStack[i]
        
        #Checks if Current frame is a blank frame
        if FrameAndCenters[1] == "None": #No Clusters
            PlotBlank(FrameName,MIP,ClusterSpyDirectory,SaveD,ImgFolder)
        else: #There are clusters
            CleanCenters = FrameAndCenters[1] #Dictionary of {ClusterLabel: XYZ}
            PlotCenters(FrameName,CleanCenters,MIP,ClusterSpyDirectory,SaveD,ImgFolder,ClusterDiameter)
            Intensities = Intensity(CleanCenters,MIP,ClusterDiameter,BufferSpacing,LocalBackgroundDiameter) #{ClusterLabel: Intensity info}
            DataForExport = ExportData(Intensities,CleanCenters)
            ExporttoCSV(FrameName,DataForExport,Pencil)  
            
        ImageGenerationValue += 1/LabelStep
        ImageGenerationProgress.set(ImageGenerationValue)
        root.update() #This line updates the GUI allowing the Progress bar to load
    Sheet.close()
    MakeMovie(ImgFolder,ClusterSpyDirectory,SaveD,FramesPerSecond)
    
#for each image passed, find the 3D centers of identified clusters
def ClusterDetectionIndividual(Stack,Bright,ContrastRatio,K):
    #Checks if Hyperstack or stack was passed through because cluster detection is done on MIP
    if len(Stack.shape) == 2: #This is a Max Intensity Projection
        MIP = Stack
        CellBlur = cv2.GaussianBlur(MIP,(GaussianFilterRange,GaussianFilterRange),0)
    elif len(Stack.shape) == 3: #This is a HyperStack
        MIP = np.max(Stack, axis = 0) #Generates Max Intensity Projection from Z stack of a given timepoint
        CellBlur = cv2.GaussianBlur(MIP,(GaussianFilterRange,GaussianFilterRange),0)
        
    #PseudoCluster applies Brightness and Contrast thresholds
    PossibleClusters = PseudoCluster(CellBlur,Bright,ContrastRatio,ClusterDiameter,BufferSpacing,LocalBackgroundDiameter)
    #Checks if there enough detected clusters given the minimum expected.
    if len(PossibleClusters)>=int(K.split("-")[0]):
        Groups,NumCluster = ClusterDetection(PossibleClusters,int(K.split("-")[0]),int(K.split("-")[1]))
        MasterList = MashingLists(PossibleClusters,Groups)
        ClusterCoordPairs = ClusterCoordPair(MasterList,NumCluster)
        Centers = ClusterCenter(ClusterCoordPairs,MIP)
        ThreeDtrack = BestZ(Centers,Stack,ClusterDiameter) #returns list of [XYZ]'s
    else: #Address no cluster scenario
        ThreeDtrack = "None"
    return ThreeDtrack
        
def DualChannelOn():
    global Dual
    Dual = True
    root.geometry("800x1000")
    global EntryFrame2
    EntryFrame2 = ctk.CTkFrame(root,corner_radius = 15)
    EntryFrame2.grid(row = 3, column = 0, padx = 15, pady = 1)
    #only the dual channel prompts need to be global so DualChannelOff can function
    global HyperStackLabel2
    global HyperStackLocate2
    global ParameterLabel2
    global Parameters2
    global BrightnessThresh2
    global LocalContrastThresh2
    global KMeansMinClusterThresh2
    global SaveDirectoryRequestOff2
    global OutputFileRequest2
    
    HyperStackLabel2 = ctk.CTkLabel(EntryFrame2, text = "Please Load TIF or OME.TIF File")
    HyperStackLabel2.grid(row = 0, column = 0, columnspan = 2)
    HyperStackLocate2 = ctk.CTkButton(EntryFrame2, text = "Select File",image = add_folder_image,command = PassTifFile2)
    HyperStackLocate2.grid(row = 1, column = 0,columnspan = 2)

    #Code asking for Parameters:
    ParameterLabel2 = ctk.CTkLabel(EntryFrame2, text = "Please Enter System Parameters Below",text_font = ("Arial",10,"bold"))
    ParameterLabel2.grid(row=4, column = 0,columnspan = 2)

    #Global Parameter handlers
    Parameters2 = ctk.CTkButton(EntryFrame2, text = "Lock in All Parameters",image = lock_icon_image, command = lambda: LockInParameters2(
                                                                                BrightnessThresh2.get(),
                                                                                LocalContrastThresh2.get(),
                                                                                KMeansMinClusterThresh2.get()))
    Parameters2.grid(row = 9, column = 0, columnspan = 2)
    
    BrightnessThreshLabel2 = ctk.CTkLabel(EntryFrame2,text = "Brightness Ratio to Background:")
    BrightnessThreshLabel2.grid(row = 5, column = 0)
    BrightnessThresh2 = ctk.CTkEntry(EntryFrame2,width = 50)
    BrightnessThresh2.grid(row = 5, column = 1,padx = 5, pady=2)
    BrightnessThresh2.insert(0,"3.6")
    
    LocalContrastThreshLabel2 = ctk.CTkLabel(EntryFrame2, text = "Pseudo-Cluster Contrast Ratio:")
    LocalContrastThreshLabel2.grid(row = 6, column = 0)
    LocalContrastThresh2 = ctk.CTkEntry(EntryFrame2,width = 50)
    LocalContrastThresh2.grid(row =6, column = 1, padx=5, pady= 2)
    LocalContrastThresh2.insert(0,"1.2")

    KMeansMinClusterThreshLabel2 = ctk.CTkLabel(EntryFrame2,text = "K-Means to Initiate:")
    KMeansMinClusterThreshLabel2.grid(row = 7, column = 0)
    KMeansMinClusterThresh2 = ctk.CTkEntry(EntryFrame2,width = 50)
    KMeansMinClusterThresh2.grid(row = 7, column = 1, padx = 5, pady=2)
    KMeansMinClusterThresh2.insert(0,"3-4")

    SaveDirectoryRequestOff2 = ctk.CTkButton(EntryFrame2, text = "Pick a Save directory",image = save_directory_image, command = lambda: CreateSaveDirectory2(OutputFileRequest2.get()),state = "disabled")
    SaveDirectoryRequestOff2.grid(row = 11, column =0, columnspan = 2, padx = 5, pady=5)

    OutputFileRequestLabel2 = ctk.CTkLabel(EntryFrame2, text = "Output Data File Name:")
    OutputFileRequestLabel2.grid(row = 10, column = 0)
    OutputFileRequest2 = ctk.CTkEntry(EntryFrame2,width = 100)
    OutputFileRequest2.grid(row = 10, column = 1, padx = 5, pady = 2)
    OutputFileRequest2.insert(0,"Results2")
    
    DualChannelAnalysisOn = ctk.CTkButton(EntryFrame, text = "Dual Channel Off", command = DualChannelOff)
    DualChannelAnalysisOn.grid(row = 2, column = 0,columnspan = 2, padx = 10, pady = 10)
        
def DualChannelOff():
    #deletes standard prompts
    global Dual
    Dual = False
    EntryFrame2.destroy()
    
    DualChannelAnalysisOn = ctk.CTkButton(EntryFrame, text = "Dual Channel On", command = DualChannelOn)
    DualChannelAnalysisOn.grid(row = 2, column = 0,columnspan = 2, padx = 10, pady = 10)
    
def DisplayImageMaster():
    DisplayProcessedImage()
    if Dual:
        DisplayProcessedImage2()

#ImageFrame exists here and Displays preview of Max Intensity Projections
def DisplayProcessedImage():
    global ImageList #Generated by opening images in save directory
    global ImageFrame  #A Master to control image display
    global SortedFiles
    global ImageDisplay
    global Horizontal
    global DisplayTitle
    #Create an ordered list of Processed Images
    fileDict = {}
    SortedFiles = []
    os.chdir(SaveDirectory)
    LabeledImages = os.listdir(ImageFolderName)
    os.chdir(ImageFolderName)
    for file in LabeledImages:
        if file.endswith(".png"):
            fileDict[int(file.split('.')[0])] = file
    for i in sorted(list(fileDict.keys())):
        SortedFiles.append(fileDict[i])
    ImageList = []
    for imgName in SortedFiles:
        ImageList.append(ImageTk.PhotoImage(Image.open(imgName).resize((DisplayResizeX, DisplayResizeY), Image.ANTIALIAS)))
        
    os.chdir(ClusterSpyDirectory)
    #Creates a frame within a window for the purpose of previewing Max Intensity Projections
    ImageFrame = ctk.CTkFrame(root)
    ImageFrame.grid(row = 2, column = 1, padx = 5, pady = 5)
    Status = ctk.CTkLabel(ImageFrame, text = "Image 1 of " + str(len(ImageList)))
    Status.grid(row = 4, column = 0, columnspan = 2)
    ImageFrameTitle = ctk.CTkLabel(ImageFrame,text = "Image Preview of Channel 1")
    ImageFrameTitle.grid(row = 0, column = 0, columnspan = 2)
    ImageDisplay = Label(ImageFrame, image = ImageList[0])
    ImageDisplay.grid(row = 2, column = 0, padx = 5, pady =5,columnspan = 2)
    Horizontal = ctk.CTkSlider(ImageFrame,from_= 1, to= len(ImageList), number_of_steps = len(ImageList)-1,width = 200, command = Move)
    Horizontal.grid(row = 3, column = 0, sticky = "E")
    Horizontal.set(1)
    
    #Creates and displays analysis plots
    
    global MaxIntensityPlotC1
    global AveIntensityPlotC1
    global MSDPlotC1
    global SizePlotC1
    global MaxIntensityPlotC2
    global AveIntensityPlotC2
    global MSDPlotC2
    global SizePlotC2
    
    #Creates the tkinter frame that the plots will go into
    SetUpCanvas() 
    
    #Creates the plots to go into the tkinter frame "PlotFrame"
    MaxIntensityPlotC1, AveIntensityPlotC1, MSDPlotC1, SizePlotC1 = SetUpData(CSVFileName,SaveDirectory,ImageFolderName,TimeStep)
    if Dual:
        MaxIntensityPlotC2, AveIntensityPlotC2, MSDPlotC2, SizePlotC2 = SetUpData(CSVFileName2,SaveDirectory2,ImageFolderName2,TimeStep)
    
    #Creates a options menu to display the various graphs
    ViewDataMenu = ctk.CTkOptionMenu(ImageFrame,variable = DataViewMenuVar,values = ['Max Intensity'
                                                                                     ,"Average Intensity"
                                                                                     ,'MSD'
                                                                                     ,"Size"]
                                 ,command = ViewData)
    ViewDataMenu.grid(row = 3, column = 1)
    ViewDataMenu.set("Max Intensity")
    
    
def Move(position):
    ImageNumber = int(position)
    DisplayTitleNew = Label(ImageFrame, text = str((int(SortedFiles[ImageNumber-1].split('.')[0])-1)*TimeStep) + " ms",bg = '#282828',fg = 'white')
    DisplayTitleNew.grid(row = 1, column = 0, columnspan = 2)
    ImageDisplay = Label(ImageFrame, image = ImageList[ImageNumber-1])
    ImageDisplay.grid(row = 2, column = 0, columnspan = 2, padx = 5, pady =5)
    Status = ctk.CTkLabel(ImageFrame, text = "Image " + str(ImageNumber) + " of " + str(len(ImageList)))
    Status.grid(row = 4, column = 0,columnspan = 2)
    
def DisplayProcessedImage2():
    global ImageList2 #Generated by opening images in save directory
    global ImageFrame2  #A Master to control image display
    global SortedFiles2
    global ImageDisplay2
    global Horizontal2
    global DisplayTitle2
    #Create an ordered list of Processed Images
    fileDict = {}
    SortedFiles2 = []
    os.chdir(SaveDirectory2)
    LabeledImages = os.listdir(ImageFolderName2)
    os.chdir(ImageFolderName2)
    for file in LabeledImages:
        if file.endswith(".png"):
            fileDict[int(file.split('.')[0])] = file
    for i in sorted(list(fileDict.keys())):
        SortedFiles2.append(fileDict[i])
    ImageList2 = []
    for imgName in SortedFiles2:
        ImageList2.append(ImageTk.PhotoImage(Image.open(imgName).resize((DisplayResizeX, DisplayResizeY), Image.ANTIALIAS)))
        
    os.chdir(ClusterSpyDirectory)
    #Creates a frame within a window for the purpose of previewing Max Intensity Projections
    ImageFrame2 = ctk.CTkFrame(root)
    ImageFrame2.grid(row = 2, column = 2, padx = 5, pady = 5)
    ImageFrameTitle2 = ctk.CTkLabel(ImageFrame2,text = "Image Preview of Channel 2")
    ImageFrameTitle2.grid(row = 0, column = 0,columnspan = 2)
    Status2 = ctk.CTkLabel(ImageFrame2, text = "Image 1 of " + str(len(ImageList2)))
    Status2.grid(row = 4, column = 0, columnspan = 2)
    ImageDisplay2 = Label(ImageFrame2, image = ImageList2[0])
    ImageDisplay2.grid(row = 2, column = 0, padx = 5, pady =5,columnspan = 2)
    Horizontal2 = ctk.CTkSlider(ImageFrame2,from_= 1, to= len(ImageList2),number_of_steps = len(ImageList)-1,width = 200, command = Move2)
    Horizontal2.grid(row = 3, column = 0,sticky = "E")
    Horizontal2.set(1)
    
    ClusterComparisonManagerButton = ctk.CTkButton(ImageFrame2,text = "Compare Dual Channel",command = ClusterComparisonManager)
    ClusterComparisonManagerButton.grid(row = 3, column = 1)
    
def Move2(position):
    ImageNumber = int(position)
    DisplayTitleNew2 = Label(ImageFrame2, text = str((int(SortedFiles2[ImageNumber-1].split('.')[0])-1)*TimeStep) + " ms",bg = '#282828',fg = 'white')
    DisplayTitleNew2.grid(row = 1, column = 0, columnspan = 2)
    ImageDisplay2 = Label(ImageFrame2, image = ImageList2[ImageNumber-1])
    ImageDisplay2.grid(row = 2, column = 0, columnspan = 2, padx = 5, pady =5)
    Status2 = ctk.CTkLabel(ImageFrame2, text = "Image " + str(ImageNumber) + " of " + str(len(ImageList2)))
    Status2.grid(row = 4, column = 0,columnspan = 2)  
    
    
def SetUpData(CSV,SaveD,FolderName,TimeStep):
    Sheet = DataForPlot(CSV,SaveD,ClusterSpyDirectory)
    Names = FindClusterNames(Sheet)
    NameTimeList = NameAndTime(Sheet, Names)
    Time = FindTimeRange(Sheet)
    
    #Set up data for Cluster Intensities
    PlotManyInt(Sheet,NameTimeList,SaveD,FolderName,ClusterSpyDirectory,ExcludeSize,"Max",TimeStep)
    PlotManyInt(Sheet,NameTimeList,SaveD,FolderName,ClusterSpyDirectory,ExcludeSize,"Average",TimeStep)
    
    #Set up data for cluster MSD
    MSDMasterList = CalculateMSDMaster(Sheet,Names,Resolution)   #[[i,MSD for i],[i+1,MSD for i+1]]
    save_MSD_sheet(SaveD,FolderName, MSDMasterList)
    Time = FindTimeRange(Sheet)
    PlotMSD(MSDMasterList,SaveD,FolderName,ClusterSpyDirectory,ExcludeSize,Time,TimeStep)
    
    #Set up data for cluster size
    PlotSize(Sheet,NameTimeList,SaveD,FolderName,ClusterSpyDirectory,ExcludeSize,Resolution,TimeStep)
    
    #Get Images AFter being saved
    os.chdir(SaveD)
    MaxIntensityPlot = ImageTk.PhotoImage(Image.open(FolderName + "_Max_IntensityPlot.png").resize((GraphResizeX,GraphResizeY), Image.ANTIALIAS))
    AveIntensityPlot = ImageTk.PhotoImage(Image.open(FolderName + "_Average_IntensityPlot.png").resize((GraphResizeX,GraphResizeY), Image.ANTIALIAS))
    MSDPlot = ImageTk.PhotoImage(Image.open(FolderName + "_MSDPlot.png").resize((GraphResizeX,GraphResizeY), Image.ANTIALIAS))
    SizePlot = ImageTk.PhotoImage(Image.open(FolderName + "_SizePlot.png").resize((GraphResizeX,GraphResizeY), Image.ANTIALIAS))
    os.chdir(ClusterSpyDirectory)
    return MaxIntensityPlot,AveIntensityPlot,MSDPlot,SizePlot
    
def SetUpCanvas():
    global PlotFrame
    PlotFrameSpan = 1
    if Dual:
        PlotFrameSpan = 2
    PlotFrame = ctk.CTkFrame(root)
    PlotFrame.grid(row = 3, column = 1,columnspan = PlotFrameSpan)
    

def ViewData(Option):
    if Option == "Max Intensity":
        MaxIntensityPlotLabel = ctk.CTkLabel(PlotFrame,image = MaxIntensityPlotC1)
        MaxIntensityPlotLabel.grid(row = 0, column = 0)
        if Dual:
            MaxIntensityPlotLabel2 = ctk.CTkLabel(PlotFrame,image = MaxIntensityPlotC2)
            MaxIntensityPlotLabel2.grid(row = 0, column = 1)
    elif Option == "Average Intensity":
        AveIntensityPlotLabel = ctk.CTkLabel(PlotFrame,image = AveIntensityPlotC1)
        AveIntensityPlotLabel.grid(row = 0, column = 0)
        if Dual:
            AveIntensityPlotLabel2 = ctk.CTkLabel(PlotFrame,image = AveIntensityPlotC2)
            AveIntensityPlotLabel2.grid(row = 0, column = 1)
    elif Option == "MSD":
        MSDPlotLabel = ctk.CTkLabel(PlotFrame, image = MSDPlotC1)
        MSDPlotLabel.grid(row = 0, column = 0)
        if Dual:
            MSDPlotLabel2 = ctk.CTkLabel(PlotFrame, image = MSDPlotC2)
            MSDPlotLabel2.grid(row = 0, column = 1)
    elif Option == "Size":
        SizePlotLabel = ctk.CTkLabel(PlotFrame, image = SizePlotC1)
        SizePlotLabel.grid(row = 0, column = 0)
        if Dual:
            SizePlotLabel2 = ctk.CTkLabel(PlotFrame, image = SizePlotC2)
            SizePlotLabel2.grid(row = 0, column = 1)
    
def ClusterComparisonManager():
    global ClusterComparisonManagerWindow
    ClusterComparisonManagerWindow = ctk.CTkToplevel()
    ClusterComparisonManagerWindow.geometry("500x400")
    ClusterComparisonManagerWindow.title("Cluster Comparison Manager")
    
    Sheet1= DataForPlot(CSVFileName, SaveDirectory, ClusterSpyDirectory)
    Channel1Names = FindClusterNames(Sheet1)
    Channel1NameTime = NameAndTime(Sheet1,Channel1Names)
    FilteredComparisonNames1 = []
    for i in Channel1NameTime:
        if len(i[1]) >= ExcludeSize:
            FilteredComparisonNames1.append(i[0])
    
    Sheet2 = DataForPlot(CSVFileName2,SaveDirectory2,ClusterSpyDirectory)
    Channel2Names = FindClusterNames(Sheet2)
    Channel2NameTime = NameAndTime(Sheet2,Channel2Names)
    FilteredComparisonNames2 = []
    for i in Channel2NameTime:
        if len(i[1]) >= ExcludeSize:
            FilteredComparisonNames2.append(i[0])
    
    
    ComparisonAskLabel = ctk.CTkLabel(ClusterComparisonManagerWindow, text = "Choose a Cluster From Each Channel to compare:", text_font = ("Arial",10,"bold"))
    ComparisonAskLabel.grid(row = 0, column = 0, columnspan = 3)
    ComparisonMenuC1 = ctk.CTkOptionMenu(ClusterComparisonManagerWindow, variable =ClusterC1, values = FilteredComparisonNames1)
    ComparisonMenuC1.set("Channel 1")
    ComparisonMenuC1.grid(row = 1, column = 0,padx = 5,pady = 5)
    ComparisonMenuC2 = ctk.CTkOptionMenu(ClusterComparisonManagerWindow, variable = ClusterC2, values = FilteredComparisonNames2)
    ComparisonMenuC2.set("Channel 2")
    ComparisonMenuC2.grid(row = 1, column = 1,padx = 5,pady = 5)
    
    CompareSelectedButton = ctk.CTkButton(ClusterComparisonManagerWindow,text = "Spacially Compare Clusters",
                                          command = lambda: CompareSelectedMaster(Sheet1,Sheet2,ComparisonMenuC1.get(),ComparisonMenuC2.get(),ImageFolderName))
    CompareSelectedButton.grid(row = 1, column = 2,padx = 5,pady = 5)
    

def CompareSelectedMaster(ClusterData1,ClusterData2,Name1,Name2,FolderName): #Chart stored in First Cluster Save Directory
    global ComparisonPlot
    ChartName, ChartDirectory = CompareSelected(ClusterData1,ClusterData2,Name1,Name2,Resolution,TimeStep,SaveDirectory,ClusterSpyDirectory,FolderName)
    os.chdir(SaveDirectory)
    os.chdir(ChartDirectory)
    ComparisonPlot = ImageTk.PhotoImage(Image.open(ChartName).resize((GraphResizeX,GraphResizeY), Image.ANTIALIAS))
    os.chdir(ClusterSpyDirectory)
    DisplayComparedClusters(ComparisonPlot)
    
def DisplayComparedClusters(ComparisonPlot):
    ComparisonFrame = ctk.CTkFrame(ClusterComparisonManagerWindow)
    ComparisonFrame.grid(row = 2, column = 0, columnspan = 3) 
    
    ComparisonPlotLabel = ctk.CTkLabel(ComparisonFrame,image = ComparisonPlot)
    ComparisonPlotLabel.grid(row = 0, column = 0)
    
    
def AdvancedOptionsMenu(): #displays the advanced options window giving the user tighter control of the program
    global AdvancedWindow
    global ClusterDiameterEntry
    global BufferSpacingEntry
    global LocalBackgroundDiameterEntry
    global ExcludeSizeEntry
    global FramesPerSecondEntry
    global ImageResolutionEntry
    global GaussianRangeEntry
    global TimeStepEntry
    
    global LoadConfigDropDown
    
    AdvancedWindow = ctk.CTkToplevel()
    AdvancedWindow.geometry("500x500")
    AdvancedWindow.title("Advanced Settings Menu")
    
    AdvancedClusterCalculationLabel = ctk.CTkLabel(AdvancedWindow,text = "Cluster Characteristics Calculation Settings",text_font = ("Arial",10,"bold"))
    AdvancedClusterCalculationLabel.grid(row = 0, column = 0, columnspan = 2)
    
    ClusterDiameterLabel = ctk.CTkLabel(AdvancedWindow, text = "Cluster Diameter in Pixels:")
    ClusterDiameterLabel.grid(row = 1,column = 0,padx = 5, pady=2)
    ClusterDiameterEntry = ctk.CTkEntry(AdvancedWindow,width = 50)
    ClusterDiameterEntry.insert(0,str(ClusterDiameter))
    ClusterDiameterEntry.grid(row = 1, column = 1,padx = 5, pady=2)
    
    BufferSpacingLabel = ctk.CTkLabel(AdvancedWindow, text = "Buffer Spacing in Pixels:")
    BufferSpacingLabel.grid(row = 2, column = 0,padx = 5, pady=2)
    BufferSpacingEntry = ctk.CTkEntry(AdvancedWindow,width = 50)
    BufferSpacingEntry.insert(0,str(BufferSpacing))
    BufferSpacingEntry.grid(row = 2, column = 1,padx = 5, pady=2)
    
    LocalBackgroundDiameterLabel = ctk.CTkLabel(AdvancedWindow, text = 'Local Background Diameter in Pixels:')
    LocalBackgroundDiameterLabel.grid(row = 3, column = 0,padx = 5, pady=2)
    LocalBackgroundDiameterEntry = ctk.CTkEntry(AdvancedWindow, width = 50)
    LocalBackgroundDiameterEntry.insert(0,str(LocalBackgroundDiameter))
    LocalBackgroundDiameterEntry.grid(row = 3, column = 1,padx = 5, pady=2)
    
    ImageResolutionLabel = ctk.CTkLabel(AdvancedWindow, text = "Image Resolution in nm/pixel:")
    ImageResolutionLabel.grid(row = 4, column = 0)
    ImageResolutionEntry = ctk.CTkEntry(AdvancedWindow, width = 50)
    ImageResolutionEntry.insert(0,str(Resolution))
    ImageResolutionEntry.grid(row = 4, column = 1)
    
    GaussianRangeLabel = ctk.CTkLabel(AdvancedWindow, text = "Gaussian Filter Diameter in Pixels:")
    GaussianRangeLabel.grid(row = 5, column = 0, padx = 5, pady = 2)
    GaussianRangeEntry = ctk.CTkEntry(AdvancedWindow, width = 50)
    GaussianRangeEntry.insert(0,str(GaussianFilterRange))
    GaussianRangeEntry.grid(row = 5, column = 1)
    
    # Control for Plots
    PlotDisplaySettingsLabel = ctk.CTkLabel(AdvancedWindow,text = "Plot Display Settings",text_font = ('Arial',10,'bold'))
    PlotDisplaySettingsLabel.grid(row = 6, column = 0, columnspan = 2, pady = 10)
    
    ExcludeSizeLabel = ctk.CTkLabel(AdvancedWindow, text = "Minimum TimeSteps for Plot:")
    ExcludeSizeLabel.grid(row = 7, column = 0,padx = 5, pady=2)
    ExcludeSizeEntry = ctk.CTkEntry(AdvancedWindow, width = 50)
    ExcludeSizeEntry.insert(0,str(ExcludeSize))
    ExcludeSizeEntry.grid(row = 7, column = 1,padx = 5, pady=2)
    
    TimeStepLabel = ctk.CTkLabel(AdvancedWindow, text = "TimeStep in miliseconds:")
    TimeStepLabel.grid(row = 8, column = 0)
    TimeStepEntry = ctk.CTkEntry(AdvancedWindow, width = 50)
    TimeStepEntry.insert(0,str(TimeStep))
    TimeStepEntry.grid(row = 8, column = 1)
    
    
    # Control for Movie
    MovieControlLabel = ctk.CTkLabel(AdvancedWindow,text = "Movie Control Settings", text_font = ('Arial',10,'bold'))
    MovieControlLabel.grid(row = 9, column = 0, columnspan = 2, pady = 10)
    
    FramesPerSecondLabel = ctk.CTkLabel(AdvancedWindow, text = "MP4 FPS:")
    FramesPerSecondLabel.grid(row = 10, column = 0,padx = 5, pady=2)
    FramesPerSecondEntry = ctk.CTkEntry(AdvancedWindow, width = 50)
    FramesPerSecondEntry.insert(0,str(FramesPerSecond))
    FramesPerSecondEntry.grid(row = 10, column = 1,padx = 5, pady=2)
                                                  
    LockAdvancedOptionsButton = ctk.CTkButton(AdvancedWindow,text = "Lock In Advanced Settings",image = lock_icon_image,
                                              command = LockAdvancedOptions)                                                      
    LockAdvancedOptionsButton.grid(row = 11, column = 0,columnspan = 2)
    
    ConfigFileNames = os.listdir(UserConfigDirectory)
    for i in range(len(ConfigFileNames)-1):
        if ConfigFileNames[i] == "Default.txt":
            ConfigFileNames.pop(i)
    LoadConfigDropDown = ctk.CTkOptionMenu(AdvancedWindow, variable = Config, values = ConfigFileNames, command = LoadConfig)
    LoadConfigDropDown.grid(row = 0, column = 2)
    LoadConfigDropDown.set(str(DefaultFile[0]))
    
    SaveConfigButton = ctk.CTkButton(AdvancedWindow,text = "Save Configurations", image = save_directory_image,state = 'disabled')
    SaveConfigButton.grid(row = 11, column = 2)
    
    MakeDefaultConfigButton = ctk.CTkButton(AdvancedWindow, text = "Make Config Default", image = Set_Default_icon_image,command = MakeDefaultConfig)
    MakeDefaultConfigButton.grid(row = 12, column = 2,pady = 5)
    
def LockAdvancedOptions(): # Locks in the users's selection for program use
    global ExcludeSize 
    global ClusterDiameter 
    global BufferSpacing 
    global LocalBackgroundDiameter 
    global FramesPerSecond 
    global Resolution
    global GaussianFilterRange
    global TimeStep
    global DefaultFile
    try: #Try to save ClusterDiameter Entry
        CDTemp = int(ClusterDiameterEntry.get())
        if CDTemp%2 == 1 and CDTemp >0:
            ClusterDiameter = CDTemp
            ClusterDiameterRequest = ctk.CTkLabel(AdvancedWindow, text = str(ClusterDiameter))
            ClusterDiameterRequest.grid(row = 1, column = 2)
        else:
            messagebox.showerror("Invalid Input","The Cluster Diameter Variable must be an Integer Odd Number")
    except:
        messagebox.showerror("Invalid Input","The Cluster Diameter Variable must be an Integer Odd Number")
        
    try: #try to save buffer spacing
        BufferSpacingTemp = int(BufferSpacingEntry.get())
        if BufferSpacingTemp >0:
            BufferSpacing = BufferSpacingTemp
            BufferSpacingRequest = ctk.CTkLabel(AdvancedWindow, text = str(BufferSpacing))
            BufferSpacingRequest.grid(row = 2, column = 2)
        else:
            messagebox.showerror("Invalid Input", "The Buffer Spacing Variable must be a Positive Integer")
    except:
        messagebox.showerror("Invalid Input","The Buffer Spacing Variable must be a Positive Integer")
    
    try: #Try to save Local Background Diameter
        LocalBackgroundDiameterTemp = int(LocalBackgroundDiameterEntry.get())
        if LocalBackgroundDiameterTemp%2 ==1:
            LocalBackgroundDiameter = LocalBackgroundDiameterTemp
            LocalBackgroundDiameterRequest = ctk.CTkLabel(AdvancedWindow, text = str(LocalBackgroundDiameter))
            LocalBackgroundDiameterRequest.grid(row = 3, column = 2)
        else:
            messagebox.showerror("Invalid Input", "The Local Background Diameter Variable must be an Integer Odd Number")
    except:
        messagebox.showerror("Invalid Input","The Local Background Diameter Variable must be an Integer Odd Number")
        
    try: #Save Microscope Resolution:
        ImageResolutionTemp = int(ImageResolutionEntry.get())
        if ImageResolutionTemp >0:
            Resolution = ImageResolutionTemp
            ImageResolutionLabel = ctk.CTkLabel(AdvancedWindow, text = str(Resolution))
            ImageResolutionLabel.grid(row = 4, column = 2)
        else:
            messagebox.showerror("Invalid Input","The Pixel Resolution should be a Positive Integer")
    except:
        messagebox.showerror("Invalid Input","The Pixel Resolution should be a Positive Integer")
        
    try: # To save Gaussian Filter Range:
        GaussianFilterRangeTemp = int(GaussianRangeEntry.get())
        if GaussianFilterRangeTemp %2 ==1 and GaussianFilterRangeTemp >0:
            GaussianFilterRange = GaussianFilterRangeTemp
            GaussianRangeLabel = ctk.CTkLabel(AdvancedWindow, text = str(GaussianFilterRange))
            GaussianRangeLabel.grid(row = 5, column = 2)
        else:
            messagebox.showerror("Invalid Input", "The Gaussian Filter Range must be a Positive Odd Integer")
    except:
        messagebox.showerror("Invalid Input", "The Gaussian Filter Range must be a Positive Odd Integer")
        
    try: #Try to save Exclude Size Variable
        ExcludeSizeTemp = int(ExcludeSizeEntry.get())
        if ExcludeSizeTemp>0:
            ExcludeSize = ExcludeSizeTemp
            ExcludeSizeRequest = ctk.CTkLabel(AdvancedWindow, text = str(ExcludeSize))
            ExcludeSizeRequest.grid(row = 7, column = 2)
        else:
            messagebox.showerror("Invalid Input", "The Exclude Size Variable must be an Positive Integer Number")
    except:
        messagebox.showerror("Invalid Input","The Exclude Size Variable must be an Positive Integer Number")
        
    try: # Try to save the Time Step time frame
        TimeStepTemp = float(TimeStepEntry.get())
        if TimeStepTemp >0:
            TimeStep = TimeStepTemp
            TimeStepLabel = ctk.CTkLabel(AdvancedWindow, text = str(TimeStep))
            TimeStepLabel.grid(row = 8, column = 2)
        else:
            messagebox.showerror("Invalid Input","The TimeStep must be a Positive Integer or Float")
    except:
        messagebox.showerror("Invalid Input","The TimeStep must be a Positive Integer or Float")
        
    try: # Try to save the FPS variable
        FramesPerSecondTemp = int(FramesPerSecondEntry.get())
        if FramesPerSecondTemp > 0:
            FramesPerSecond = FramesPerSecondTemp
            FramesPerSecondRequest = ctk.CTkLabel(AdvancedWindow, text = str(FramesPerSecond))
            FramesPerSecondRequest.grid(row = 10, column = 2)
        else:
            messagebox.showerror("Invalid Input","The FPS Vairable must be a Positive Integer Number")
    except:
        messagebox.showerror("Invalid Input","The FPS Vairable must be a Positive Integer Number")
        
    SaveConfigButton = ctk.CTkButton(AdvancedWindow,text = "Save Configurations", image = save_directory_image,command = SaveConfig)
    SaveConfigButton.grid(row = 11, column = 2)
        
def SaveConfig(): #Creates a new config file with current configurations
    os.chdir(UserConfigDirectory)
    LengthOfDirectory = len(os.listdir(os.getcwd()))-1
    f = open("Config" + str(LengthOfDirectory+1) +".txt",'w+')  
    f.write("Cluster Diameter =" +str(ClusterDiameter) + '\n')
    f.write("Buffer Spacing =" + str(BufferSpacing) + '\n')
    f.write("Local Background Diameter =" + str(LocalBackgroundDiameter) + '\n')
    f.write("Image Resolution in nm/pixel =" + str(Resolution) + '\n')
    f.write("Gaussian Filter Diameter =" + str(GaussianFilterRange) + '\n')
    f.write("Minimum TimeSteps for Plot =" + str(ExcludeSize) + '\n')
    f.write("Time Step in ms =" + str(TimeStep)+ '\n')
    f.write("MP4 FPS =" + str(FramesPerSecond) + '\n')
    f.close()
    os.chdir(ClusterSpyDirectory)
    
    ConfigFileNames = os.listdir(UserConfigDirectory)
    for i in range(len(ConfigFileNames)-1): #Ignores the default file for selecting the default config file
        if ConfigFileNames[i] == "Default.txt":
            ConfigFileNames.pop(i)
    LoadConfigDropDown = ctk.CTkOptionMenu(AdvancedWindow, variable = Config, values = ConfigFileNames, command = LoadConfig)
    LoadConfigDropDown.set("Config" + str(LengthOfDirectory+1)+".txt")
    LoadConfigDropDown.grid(row = 0, column = 2)
    
def LoadConfig(Option): #Reads information from the config file and stores them into their proper variables.
    global ExcludeSize 
    global ClusterDiameter 
    global BufferSpacing 
    global LocalBackgroundDiameter 
    global FramesPerSecond 
    global Resolution
    global GaussianFilterRange
    global TimeStep
    
    os.chdir(UserConfigDirectory)
    f = open(Option,'r')
    Configs = f.readlines()
    f.close()
    
    ClusterDiameter = int(Configs[0].split('=')[1])
    ClusterDiameterRequest = ctk.CTkLabel(AdvancedWindow, text = str(ClusterDiameter))
    ClusterDiameterRequest.grid(row = 1, column = 2)
    ClusterDiameterEntry.delete(0,'end')
    ClusterDiameterEntry.insert(0,str(ClusterDiameter))
    
    BufferSpacing = int(Configs[1].split('=')[1])
    BufferSpacingRequest = ctk.CTkLabel(AdvancedWindow, text = str(BufferSpacing))
    BufferSpacingRequest.grid(row = 2, column = 2)
    BufferSpacingEntry.delete(0,'end')
    BufferSpacingEntry.insert(0,str(BufferSpacing))
    
    LocalBackgroundDiameter = int(Configs[2].split('=')[1])
    LocalBackgroundDiameterRequest = ctk.CTkLabel(AdvancedWindow, text = str(LocalBackgroundDiameter))
    LocalBackgroundDiameterRequest.grid(row = 3, column = 2)
    LocalBackgroundDiameterEntry.delete(0,'end')
    LocalBackgroundDiameterEntry.insert(0,str(LocalBackgroundDiameter))
    
    Resolution = int(Configs[3].split('=')[1])
    ImageResolutionLabel = ctk.CTkLabel(AdvancedWindow, text = str(Resolution))
    ImageResolutionLabel.grid(row = 4, column = 2)
    ImageResolutionEntry.delete(0,'end')
    ImageResolutionEntry.insert(0,str(Resolution))
    
    GaussianFilterRange = int(Configs[4].split('=')[1])
    GaussianRangeLabel = ctk.CTkLabel(AdvancedWindow, text = str(GaussianFilterRange))
    GaussianRangeLabel.grid(row = 5, column = 2)
    GaussianRangeEntry.delete(0,'end')
    GaussianRangeEntry.insert(0,str(GaussianFilterRange))
    
    ExcludeSize = int(Configs[5].split('=')[1])
    ExcludeSizeRequest = ctk.CTkLabel(AdvancedWindow, text = str(ExcludeSize))
    ExcludeSizeRequest.grid(row = 7, column = 2)
    ExcludeSizeEntry.delete(0,'end')
    ExcludeSizeEntry.insert(0,str(ExcludeSize))
    
    TimeStep = float(Configs[6].split('=')[1])
    TimeStepLabel = ctk.CTkLabel(AdvancedWindow, text = str(TimeStep))
    TimeStepLabel.grid(row = 8, column = 2)
    TimeStepEntry.delete(0,'end')
    TimeStepEntry.insert(0,str(TimeStep))
    
    FramesPerSecond = int(Configs[7].split('=')[1])
    FramesPerSecondRequest = ctk.CTkLabel(AdvancedWindow, text = str(FramesPerSecond))
    FramesPerSecondRequest.grid(row =10 , column = 2)
    FramesPerSecondEntry.delete(0,'end')
    FramesPerSecondEntry.insert(0,str(FramesPerSecond))
    
    os.chdir(ClusterSpyDirectory)
        
def SetAdvancedParameters():
    #Advanced Setting parameters
    global ExcludeSize 
    global ClusterDiameter 
    global BufferSpacing 
    global LocalBackgroundDiameter 
    global FramesPerSecond 
    global Resolution
    global GaussianFilterRange
    global TimeStep
    global DefaultFile
    
    os.chdir(UserConfigDirectory)
    f = open("Default.txt",'r')
    DefaultFile = f.readlines()
    # print(DefaultFile[0])
    f.close()
    f = open(DefaultFile[0],'r')
    Configs = f.readlines()
    f.close()
    
    ClusterDiameter = int(Configs[0].split('=')[1])
    BufferSpacing = int(Configs[1].split('=')[1])
    LocalBackgroundDiameter = int(Configs[2].split('=')[1])
    Resolution = int(Configs[3].split('=')[1])
    GaussianFilterRange = int(Configs[4].split('=')[1])
    ExcludeSize = int(Configs[5].split('=')[1])
    TimeStep = float(Configs[6].split('=')[1])
    FramesPerSecond = int(Configs[7].split('=')[1])
    
    os.chdir(ClusterSpyDirectory)
    
def MakeDefaultConfig():
    os.chdir(UserConfigDirectory)
    f = open("Default.txt",'w+')
    f.write(str(LoadConfigDropDown.get()))
    f.close()
    os.chdir(ClusterSpyDirectory)
    messagebox.showinfo("Default Advanced Configuration Saved", str(LoadConfigDropDown.get()) + " became the new Default Advanced Configuration")

#Custom GUi appearance and theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"
root = ctk.CTk()
root.title("ClusterSpy")
root.geometry("800x800")

image_size = 20
# add_folder_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/Miku.png").resize((image_size*7, image_size*7), Image.ANTIALIAS))
add_folder_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/add-folder.png").resize((image_size, image_size), Image.ANTIALIAS))
save_directory_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/Save-Icon.png").resize((image_size, image_size), Image.ANTIALIAS))
lock_icon_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/Lock-Icon.png").resize((image_size, image_size), Image.ANTIALIAS))
clusterSpy_icon_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/ClusterSpy.png").resize((40, 40), Image.ANTIALIAS))
Advanced_settings_icon_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/Gear.png").resize((30,30), Image.ANTIALIAS))
Set_Default_icon_image = ImageTk.PhotoImage(Image.open(ClusterSpyDirectory + r"/Program_Icons/House-Default.png").resize((image_size,image_size), Image.ANTIALIAS))
Program_Icon = PhotoImage(file = ClusterSpyDirectory + r"/Program_Icons/ClusterSpy.png")

root.iconphoto(True,Program_Icon)

DataViewMenuVar = StringVar()
Config = StringVar()
ClusterC1 = StringVar()
ClusterC2 = StringVar()

SetAdvancedParameters()  #Sets up advanced parameters  

#Code to Control User Input
EntryFrame = ctk.CTkFrame(root,corner_radius = 15)
EntryFrame.grid(row = 0, column = 0, padx = 15, pady = 1,rowspan = 3)

#Code to ask for directory of Max Intensity Folder
HyperStackLabel = ctk.CTkLabel(EntryFrame, text = "Please Load TIF or OME.TIF File")
HyperStackLabel.grid(row = 0, column = 0, columnspan = 2)
HyperStackLocate = ctk.CTkButton(EntryFrame, text = "Select File",image = add_folder_image,command = PassTifFile)
HyperStackLocate.grid(row = 1, column = 0, columnspan = 2)

#Code asking for Parameters:
ParameterLabel = ctk.CTkLabel(EntryFrame, text = "Please Enter System Parameters Below",text_font = ("Arial",10,"bold"))
ParameterLabel.grid(row=4, column = 0, columnspan = 2)

#Global Parameter handlers
Parameters = ctk.CTkButton(EntryFrame, text = "Lock in All Parameters", image = lock_icon_image, command = lambda: LockInParameters(
                                                                            BrightnessThresh.get(),
                                                                            LocalContrastThresh.get(),
                                                                            KMeansMinClusterThresh.get()))
Parameters.grid(row = 9, column = 0, columnspan = 2)

BrightnessThreshLabel = ctk.CTkLabel(EntryFrame,text ="Brightness Ratio to Background:")
BrightnessThreshLabel.grid(row = 5, column = 0)
BrightnessThresh = ctk.CTkEntry(EntryFrame,width = 50)
BrightnessThresh.grid(row = 5, column = 1,padx = 5, pady=2)
BrightnessThresh.insert(0,"3.6")

LocalContrastThreshLabel = ctk.CTkLabel(EntryFrame, text = "Pseudo-Cluster Contrast Ratio:")
LocalContrastThreshLabel.grid(row = 6, column = 0)
LocalContrastThresh = ctk.CTkEntry(EntryFrame,width = 50)
LocalContrastThresh.grid(row =6, column = 1, padx=5, pady= 2)
LocalContrastThresh.insert(0, "1.2")

KMeansMinClusterThreshLabel = ctk.CTkLabel(EntryFrame,text = "K-Means to Initiate:")
KMeansMinClusterThreshLabel.grid(row = 7, column = 0)
KMeansMinClusterThresh = ctk.CTkEntry(EntryFrame,width = 50)
KMeansMinClusterThresh.grid(row = 7, column = 1, padx = 5, pady=2)
KMeansMinClusterThresh.insert(0,"3-4")

SaveDirectoryRequest = ctk.CTkButton(EntryFrame, text = "Pick a Save directory", image = save_directory_image, command = lambda: CreateSaveDirectory(OutputFileRequest.get()),state = "disabled")
SaveDirectoryRequest.grid(row = 11, column =0, columnspan = 2,padx = 5, pady=5)

OutputFileRequestLabel = ctk.CTkLabel(EntryFrame, text = "Output Data File Name:")
OutputFileRequestLabel.grid(row = 10, column = 0)
OutputFileRequest = ctk.CTkEntry(EntryFrame, width = 100)
OutputFileRequest.grid(row = 10, column = 1, padx = 1, pady = 2)
OutputFileRequest.insert(0,"Results")

#Button that runs the ClusterSpy backend
MasterRun = ctk.CTkButton(EntryFrame, text = "Run Analysis",image = clusterSpy_icon_image,text_color = "Black" , fg_color = "red",hover_color = "#FFAFAF",command = MasterButton)
MasterRun.grid(row =0, column = 2, columnspan = 2)

AdvancedOptionsButton = ctk.CTkButton(EntryFrame, text = "Advanced Settings", image = Advanced_settings_icon_image, command = AdvancedOptionsMenu)
AdvancedOptionsButton.grid(row = 1, column = 2, columnspan = 2,pady = 3)

DualChannelAnalysisOn = ctk.CTkButton(EntryFrame, text = "Dual Channel On", command = DualChannelOn)
DualChannelAnalysisOn.grid(row = 2, column = 0, columnspan = 2, padx = 10, pady = 10)

DisplayProcessedImageButton = ctk.CTkButton(EntryFrame, text = "Display", command = DisplayImageMaster, state = "disabled")
DisplayProcessedImageButton.grid(row = 2, column = 2)

root.mainloop()