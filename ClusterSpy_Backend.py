import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import cv2
import os
import pandas as pd
import statistics as stat
os.environ["OMP_NUM_THREADS"] = '1'

#Used to calculate image average intensity
def AveImPixel(ImageArray):
   rows,cols = ImageArray.shape 
   PixVal = []
   #Adds pixel values to dictionary for threshold counting
   for i in range(rows):
       for j in range(cols):
           PixVal.append(ImageArray[i,j])
   return sum(PixVal)/len(PixVal)
                
#Filters pixels with the pseudo-cluster threshold
def PseudoCluster(BlurPixelData,Brightness,LocalContrastvar,ClusterDiameter,BufferSpacing,LocalBackgroundDiameter):
    #Applies Brightness threshold to individual pixels, then applies Local Contrast.
    # Candidates who pass are returned a Candidates variable
    ClusterSizeL = -(ClusterDiameter//2)
    ClusterSizeR = ClusterDiameter + ClusterSizeL
    LocalBackgroundL = -(LocalBackgroundDiameter//2)
    LocalBackgroundR = LocalBackgroundDiameter + LocalBackgroundL
    
    rows,cols = BlurPixelData.shape
    AveBackground = AveImPixel(BlurPixelData)
    Candidates = []
    for i in range(rows):
        for j in range(cols):
            if BlurPixelData[i,j] > AveBackground*(Brightness):
                LocalBackground= []
                for width in range(LocalBackgroundL,LocalBackgroundR):
                    for length in range(LocalBackgroundL,LocalBackgroundR):
                        if not((width in range(ClusterSizeL-BufferSpacing,ClusterSizeR+BufferSpacing)) and 
                                (length in range(ClusterSizeL-BufferSpacing,ClusterSizeR+BufferSpacing))):
                            if (length+i in range(rows) and 
                                    width+j in range(cols)):
                                LocalBackground.append(BlurPixelData[length+i,width+j])
                LocalAve = sum(LocalBackground)/len(LocalBackground)
                if BlurPixelData[i,j] >= LocalAve*LocalContrastvar:
                    Candidates.append([j,i])
    return Candidates

#Binary selection of probable cluster pixels
def FalseImage(PosClusters,RawImage): 
    FalseIm = np.copy(RawImage)
    rows,cols = FalseIm.shape
    white = np.amax(FalseIm)*30
    for i in range(rows):
        for j in range(cols): #makes screen black for coloring with white
            FalseIm[i,j] = 0 
    for coords in PosClusters:
        FalseIm[coords[1],coords[0]] = white #Highlights posible cluster pixels
    return FalseIm

#using K meams clustering to define clusters
def ClusterDetection(PossibleClusters,MinCluster,MaxCluster): 
    k_range = range(MinCluster,MaxCluster)  
    SSE = [] #Keeps track of the SSE for each K means ran
    PreCalcKm = {}
    VarDif = 0
    NumClusters = MinCluster
    for k in k_range:
        if k <= len(PossibleClusters):
            km = KMeans(n_clusters = k,n_init= 5, max_iter=50, algorithm= 'elkan')
            km.fit(PossibleClusters)
            SSE.append(km.inertia_)
            PreCalcKm[k] = km
    for i in range(len(SSE)-1):
        if SSE[i+1] ==0:
            break
        if (SSE[i]/SSE[i+1]) > VarDif:
            VarDif = (SSE[i]/SSE[i+1])
            NumClusters += i+1
    
    Selected = PreCalcKm[NumClusters]
    Groupings = Selected.fit_predict(PossibleClusters)
    return Groupings,NumClusters

#matches cluster identity with location
def MashingLists(Location,Group): 
    Groupings = np.copy(Group)
    Master = []
    for i in range(len(Location)):
        Master.append([Location[i],Groupings[i]])
    return Master

#Takes the master list and returns a dictionary of cluster identity + all pixels in that cluster
def ClusterCoordPair(MList,NumCluster): 
    ClusterCoordDict = {}
    for i in range(NumCluster):
        coordmap = []
        for j in MList:
            if j[1] == i:
                coordmap.append(j[0])
        ClusterCoordDict[i+1] = coordmap
    return ClusterCoordDict

#Takes a dictionary of clusters and raw data to produce centers for each cluster
def ClusterCenter(ClusterCoordPairs,imageP): 
    CenterofClusters = {}
    for Cluster in ClusterCoordPairs.keys():
        PixelSize = 0
        xCenter = 0
        yCenter = 0
        PixelSum = 0
        for coords in ClusterCoordPairs[Cluster]:
            xCenter += imageP[coords[1],coords[0]]*coords[0]
            yCenter += imageP[coords[1],coords[0]]*coords[1]
            PixelSum += imageP[coords[1],coords[0]]
            PixelSize +=1
        xCenter = round(xCenter/PixelSum,2)
        yCenter = round(yCenter/PixelSum,2)
        CenterofClusters[Cluster] = [[yCenter,xCenter],PixelSize]
    return CenterofClusters

#calculates cluster intensity as well as background intesntiy
def Intensity (Centers,ImPixels,ClusterDiameter,BufferSpacing,LocalBackgroundDiameter):
    ClusterSizeL = -(ClusterDiameter//2)
    ClusterSizeR = ClusterDiameter + ClusterSizeL
    LocalBackgroundL = -(LocalBackgroundDiameter//2)
    LocalBackgroundR = LocalBackgroundDiameter + LocalBackgroundL
    IntensityDict = {}
    rows,cols = ImPixels.shape
    for center in Centers.keys():
        ClusterIntensity = 0
        BackgroundIntensity = 0
        xfloor = math.floor(Centers[center][0][0])
        yfloor = math.floor(Centers[center][0][1])
        MaxP = 0
        ClusterSize = 0
        BackgroundSize = 0
        for i in range(ClusterSizeL,ClusterSizeR):
            for j in range(ClusterSizeL,ClusterSizeR):
                if xfloor+j in range(cols) and yfloor+i in range(rows):
                    ClusterIntensity += ImPixels[yfloor+i,xfloor+j]
                    ClusterSize +=1
                    if ImPixels[yfloor+i,xfloor+j] > MaxP:
                        MaxP = ImPixels[yfloor+i,xfloor+j]
        for i in range(LocalBackgroundL,LocalBackgroundR):
            for j in range(LocalBackgroundL,LocalBackgroundR):
                if ( i in range(ClusterSizeL-BufferSpacing,ClusterSizeR+BufferSpacing) and 
                    j in range(ClusterSizeL-BufferSpacing,ClusterSizeR+BufferSpacing)):
                    break
                else:
                    if xfloor+j in range(cols) and yfloor+i in range(rows):
                        BackgroundIntensity+= ImPixels[yfloor+i,xfloor+j]
                        BackgroundSize +=1
        IntensityDict[center] = [MaxP, round(ClusterIntensity/ClusterSize,2),round(BackgroundIntensity/BackgroundSize,2)]
    return IntensityDict 

#Using centers derived from max intensity plots, finds brightest z layer of cluster
def BestZ(Cen,Stack,ClusterDiameter):  
    XYZList = [] #Stores XYX in a list
    ClusterSizeL = -(ClusterDiameter//2)
    ClusterSizeR = ClusterDiameter + ClusterSizeL
    for Cluster in Cen.keys():
        position= Cen[Cluster] #Grabs the X,Y and the clsuter size, but cluster size is just passed through
        y = math.floor(position[0][0])
        x = math.floor(position[0][1])
        if len(Stack.shape) == 2: #Max Intensity Projection was passed
            MaxZ = 1
        elif len(Stack.shape) == 3: #HyperStack was passed
            MaxI = 0
            MaxZ = 1
            for ZSlice in range(len(Stack)):
                ClusterIntensity = 0
                ClusterSize = 0
                rows,cols = Stack[ZSlice].shape
                for i in range(ClusterSizeL,ClusterSizeR):
                    for j in range(ClusterSizeL,ClusterSizeR):
                        if (x+j in range(cols) and y+i in range(rows)):
                            ClusterIntensity += Stack[ZSlice][y+i,x+j]
                            ClusterSize +=1
                if (ClusterIntensity/ClusterSize) > MaxI:
                    MaxI = ClusterIntensity/ClusterSize
                    MaxZ = ZSlice
            
        XYZList.append([[position[0][1],position[0][0],MaxZ],position[1]])  #([X,Y,Z],Size of cluster)
    return XYZList #List of [XYZ]'s

#Takes the centers, opens up max intensity plots and overlays a circle and label on a new png
def PlotCenters(T,Centers,Cell,Pdirectory,SavedImageDir,SavedFolderName,ClusterDiameter):
    os.chdir(SavedImageDir)
    os.chdir(SavedFolderName)
    rowcent = []
    colcent = []
    Label = []
    for center in Centers.keys():
        overlap = False
        for i in range(len(rowcent)):
            Distance = ((Centers[center][0][0] - rowcent[i])**2 + (Centers[center][0][1]-colcent[i])**2)**0.5
            if Distance <=ClusterDiameter:
                overlap = True
        if not overlap:
            Label.append(center)
            rowcent.append(Centers[center][0][0])
            colcent.append(Centers[center][0][1])
    plt.clf()
    plt.title(str(T+1))
    plt.imshow(Cell,cmap = "gray")
    plt.axis('off')
    plt.scatter(rowcent,colcent, facecolor = 'none',edgecolor = 'r', s = 100)
    for i in range(len(Label)):
        y,x = rowcent[i],colcent[i]
        plt.annotate(str(Label[i]),(y,x),xytext = (y+5,x+5), color = 'white')
        
    plt.savefig(str(T+1) + ".png")
    plt.close()
    os.chdir(Pdirectory)
    
def PlotBlank(T,Cell,Pdirectory,SavedImageDir,SavedFolderName):
    os.chdir(SavedImageDir)
    os.chdir(SavedFolderName)
    #Plots a blank image for continuity
    plt.clf()
    plt.title(str(T+1))
    plt.imshow(Cell,cmap = "gray")
    plt.axis('off')
    plt.savefig(str(T+1) + ".png")
    plt.close()
    os.chdir(Pdirectory)

#This function uses a global variable CenterMaster from the front end to find similar clusters between frames
def TrackClusterBetweenFrames(CenterMaster,ClusterDiameter):
    TrackedClustersMaster = [] #Stores Center information for all frames
    MasterFrameID = 1
    # print(len(CenterMaster))
    for i in CenterMaster.keys():
        ValidCheck = False
        TrackedClusters = {} #Stores center information as a dictionary with label and coords
        FrameTime = i
        Frame = CenterMaster[i]
        try:  #Checks if you can access one frame behind
            TrackedClusterBehind = TrackedClustersMaster[-1]
            FrameBehindTime = TrackedClusterBehind[0]
            FrameBehind = TrackedClusterBehind[1]
            if FrameBehindTime == FrameTime -1:
                ValidCheck =True
        except:
            pass
        if Frame == "None": #Checks for a No cluster found frame
            TrackedClustersMaster.append([FrameTime,"None"])
        else: #There are clusters in the frame
            if ValidCheck ==True: #Can we properly reference the frame behind?
                for center in Frame:
                    SameClusterNextFrame =False
                    if FrameBehind == "None":
                        TrackedClusters[MasterFrameID] = center
                        MasterFrameID +=1
                    else:
                        for ID in FrameBehind.keys():
                            centerbehind = FrameBehind[ID]
                            Distance = ((center[0][0]-centerbehind[0][0])**2 + (center[0][1]-centerbehind[0][1])**2)**0.5
                            if Distance <=ClusterDiameter: # looks for a probable cluster in a similar place
                                TrackedClusters[ID] = center
                                SameClusterNextFrame =True
                                break
                        if not SameClusterNextFrame:    
                            TrackedClusters[MasterFrameID] = center
                            MasterFrameID +=1
            else: #should be base case aka first frame
                for center in Frame:
                    TrackedClusters[MasterFrameID] = center
                    MasterFrameID +=1
                    
            #once all clusters have been given a label append that to a master list below 
            TrackedClustersMaster.append([FrameTime,TrackedClusters]) 
        # print(TrackedClusters)
    return TrackedClustersMaster

#Takes all the pngs generated for identified clusters and creates an MP4 movie at 4 frames a second
def MakeMovie(SavedImageDir,Pdirectory,SaveD,FramesPerSecond):
    os.chdir(SaveD)
    ImageList = os.listdir(SavedImageDir)
    os.chdir(SavedImageDir)
    fileDict = {}
    SortedFiles = []
    exampleImage = cv2.imread(ImageList[0])
    height,width,layers = exampleImage.shape
    size = (width,height)
    for i in range(len(ImageList)):
        fileDict[int(ImageList[i].split('.')[0])] = ImageList[i]
    for i in sorted(list(fileDict.keys())):
        SortedFiles.append(fileDict[i])
    out = cv2.VideoWriter("Processed Video.mp4",cv2.VideoWriter_fourcc(*"mp4v"),FramesPerSecond,size)
    for filename in SortedFiles:
        img = cv2.imread(filename)
        out.write(img)
    out.release()
    os.chdir(Pdirectory)

#Takes intensities as well as centers and puts it into a format ready to be exported to csv
def ExportData(Intensity,XYZ):
    MergedDict = {}
    for i in sorted(Intensity.keys()):
        MergedDict[i] = XYZ[i][0] + [XYZ[i][1]] + Intensity[i]
    return MergedDict

#Opens the user created csv and transfers data
def ExporttoCSV(TimeStamp,Data,Pencil):
    for i in Data.keys():
        Masterrow = [TimeStamp]
        Masterrow.append(i)
        for data in Data[i]:
            Masterrow.append(data)
        Pencil.writerow(Masterrow)

def save_MSD_sheet(save_path, folder, MSDMasterList):
    df = pd.DataFrame(MSDMasterList)
    colnames = ['Cluster '+str(col) for col in df[0]]
    d = {}
    for c, col in enumerate(colnames):
        d[col] = MSDMasterList[c][1]
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()
    # print('Saving MSD to csv ...')
    df.to_csv(os.path.join(save_path, folder+'_MSD.csv'), index=False)
    return
      
#Creates panda dataframe from excel sheet
def DataForPlot(file,saveD,Pdirectory): 
    os.chdir(saveD)
    ClusterStats = pd.read_csv(file)
    os.chdir(Pdirectory)
    return ClusterStats

def FindClusterNames(ClusterData):
    ClusterNames = []
    for i in ClusterData.iloc[:,1]:
        if i in ClusterNames:
            pass
        else:
            ClusterNames.append(i)
    return ClusterNames

def NameAndTime(ClusterData,Names):
    NameTimeList = []
    for i in Names:
        NameTimeList.append([i,ClusterData.loc[ClusterData["Cluster Name"] == i,"TimeStamp"]])
    return NameTimeList

def FindTimeRange(ClusterData):
    Time = []
    for i in ClusterData.iloc[:,0]:
        if i in Time:
            pass
        else:
            Time.append(i)
    return Time

def PlotManyInt(ClusterData,NameTimeList,SDirectory,ImageFolderName,ClusterSpyDirectory,ExcludeSize,IntensityType,TimeStep):
    for i in NameTimeList:
        if len(i[1])>ExcludeSize:
            plt.plot([j*TimeStep for j in i[1]],ClusterData.loc[ClusterData["Cluster Name"] == i[0],IntensityType],marker = 'o',label = i[0])
    plt.title(str(IntensityType) + " Intensity")
    plt.xlabel("Time in ms")
    plt.ylabel("Intensity")
    plt.tick_params(top = False, bottom = True, left = True, right = False)
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    os.chdir(SDirectory)
    plt.savefig(ImageFolderName +"_" + str(IntensityType) + "_IntensityPlot.png")
    plt.close()
    os.chdir(ClusterSpyDirectory)
    
    
def CalculateMSDMaster(Sheet,Names,Resolution):
    MSDMasterList = []
    for i in Names:
        df = Sheet.loc[Sheet["Cluster Name"] == i,['X','Y']] #Holds the x and y of a given cluster
        df = df.reset_index(drop = True)
        MSDClusterMean = [] #Holds all MSD for a given cluster
        MSDClusterSEM = [] #Holds all the MSD SEM for a given cluster
        for dt in range(1,df.shape[0]):
            MSDdt = []
            for j in range(0,df.shape[0]):
                if j + dt > df.shape[0]-1:
                    break
                else:  #takes data in pixel length and computes MSD in um^2
                    MSDdt.append((((df.iloc[j+dt,0] - df.iloc[j,0])*Resolution)**2 +  ((df.iloc[j+dt,1] - df.iloc[j,1])*Resolution)**2)*(10**-6))
            if len(MSDdt)<=1:
                MSDClusterSEM.append(MSDdt[0])
            else:
                MSDClusterSEM.append((stat.stdev(MSDdt))/(len(MSDdt))**0.5)
            MSDClusterMean.append(stat.mean(MSDdt))
        MSDMasterList.append([i,MSDClusterMean,MSDClusterSEM])
    return MSDMasterList

def PlotMSD(MSDMasterList,SaveD,ImageFolderName,ClusterSpyDirectory,ExcludeSize,Time,TimeStep):
    for i in MSDMasterList:
        if len(i[1])>ExcludeSize:
            TimeSpan = list(np.array(range(0,len(i[1]))))
            plt.errorbar([j*TimeStep/1000 for j in TimeSpan],i[1],yerr=i[2],marker = 'o',label = i[0])
    # plt.plot([0.001,.01],[0.004,.04],marker = 'o',label = "y=x")
    plt.title('MSD Plot')
    plt.xlabel('Timelag (s)')
    plt.ylabel("Mean Squared Displacement (um^2)")
    plt.tick_params(top = False, bottom = True, left = True, right = False)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log',base=10)
    plt.xscale('log',base=10)
    plt.tight_layout()
    os.chdir(SaveD)
    plt.savefig(ImageFolderName + "_MSDPlot.png")
    plt.close()
    os.chdir(ClusterSpyDirectory)
    
def PlotSize(ClusterData,NameTimeList,SDirectory,ImageFolderName,ClusterSpyDirectory,ExcludeSize,Resolution,TimeStep):
    for i in NameTimeList:
        if len(i[1])>ExcludeSize:
            plt.plot([j*TimeStep for j in i[1]],(((ClusterData.loc[ClusterData["Cluster Name"] == i[0],"ClusterSize"])/np.pi)**0.5)*Resolution,marker = 'o',label = i[0])
    plt.title("Cluster Size Over Time")
    plt.xlabel("Time in ms")
    plt.ylabel("Approximate Cluster Radius in nm")
    plt.tick_params(top = False, bottom = True, left = True, right = False)
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    os.chdir(SDirectory)
    plt.savefig(ImageFolderName + "_SizePlot.png")
    plt.close()
    os.chdir(ClusterSpyDirectory)
    
def CompareSelected(ClusterData1,ClusterData2,Name1,Name2,Resolution,TimeStep,SDirectory,ClusterSpyDirectory,FolderName):
    #this segment of code checks for a Comparison_Chart directory, where charts generated will be stored.
    os.chdir(SDirectory)
    ComparisonDirectory = "Comparison_Charts"
    if not os.path.isdir(ComparisonDirectory): #Does the directory already exist?
        os.makedirs(ComparisonDirectory)
    os.chdir(ComparisonDirectory)
    
    #Subfunction Called to specifically return a list of differences in position  
    DifferenceList,CommonMin,CommonMax = FindSpacialDifference(ClusterData1,ClusterData2,Name1,Name2,Resolution)
        
    #This part of the code plots the actual chart
    TimeAxis = [i*TimeStep for i in range(CommonMin,CommonMax+1)]
    SavedPlot = PlotComparison(DifferenceList,Name1,Name2,TimeAxis,FolderName)
    os.chdir(ClusterSpyDirectory)
    
    #Returns the Chart and the Directory
    return(SavedPlot,ComparisonDirectory)

#This section of code is responsible for spitting out a difference list between the common overlap of 2 clusters 
def FindSpacialDifference(ClusterData1,ClusterData2,Name1,Name2,Resolution):  
    NameAndTime1 = NameAndTime(ClusterData1,[Name1])
    NameAndTime2 = NameAndTime(ClusterData2,[Name2])
    CommonTime = []
    for i in NameAndTime1[0][1]:
        if i in NameAndTime2[0][1]:
            CommonTime.append(i)
    PositionName1 = ClusterData1.loc[ClusterData1["Cluster Name"] == Name1,['TimeStamp','X','Y']]
    PositionName2 = ClusterData2.loc[ClusterData2["Cluster Name"] == Name2,['TimeStamp','X','Y']]
    minName1 = min(PositionName1['TimeStamp'])
    maxName1 = max(PositionName1['TimeStamp'])
    minName2 = min(PositionName2['TimeStamp'])
    maxName2 = max(PositionName2["TimeStamp"])
    CommonMin = max([minName1,minName2])
    CommonMax = min([maxName1,maxName2])
    DifferenceList=[] #Will hold differences from commonmin to commonmax+1
    for i in range(CommonMin,CommonMax+1):
        CoordName1 = PositionName1.loc[PositionName1["TimeStamp"] == i,['X','Y']]
        CoordName2 = PositionName2.loc[PositionName2["TimeStamp"] == i,['X','Y']]
        X1 = float(CoordName1.iloc[0,0])
        Y1 = float(CoordName1.iloc[0,1])
        X2 = float(CoordName2.iloc[0,0])
        Y2 = float(CoordName2.iloc[0,1])
        DifferenceList.append((((X1-X2)**2 + (Y1-Y2)**2)**0.5)*Resolution)
    return DifferenceList,CommonMin,CommonMax
    
def PlotComparison(DifferenceList,Name1,Name2,TimeAxis,FolderName):
    title = "Distance Between Centers of C1: " + str(Name1) +" & C2: " + str(Name2)
    plt.title(title)
    plt.xlabel("Time in ms")
    plt.ylabel("Distance Between Clusters in nm")
    plt.plot(TimeAxis,DifferenceList,marker = 'o')
    plt.tight_layout()
    ComparisonSaveFileName = str(FolderName) + "_" + str(Name1) + "&" + str(Name2) + "_Comparison.png"
    plt.savefig(ComparisonSaveFileName)
    plt.close()
    return ComparisonSaveFileName
    