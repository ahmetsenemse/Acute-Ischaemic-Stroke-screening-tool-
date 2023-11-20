import os
import numpy as np
import fourierseries as fr
import artery_area as ata
import pyvista
import scipy.io as sio


series=sio.loadmat('database.mat')
reader = pyvista.read('C:/Users/ahmet.sen/Downloads/ActualBrain/ImageBasedModel/Vasculature(LV0)/Artery.vtp')
a=reader.points
b=reader.lines.reshape(int(17970/3),3)[:,1:3]
rad=reader.active_scalars
b[1187,:]=[-1,-1]
b[2242,:]=[-1,-1]
b[3873,:]=[-1,-1]
b[5306,:]=[-1,-1]
b[5503,:]=[-1,-1]
b[5779,:]=[-1,-1]

i=0
while i<len(a):
    interp=np.intersect1d(np.where(a[:,0]==a[i,0])[0],np.where(a[:,1]==a[i,1])[0])
    interp1=np.intersect1d(interp,np.where(a[:,2]==a[i,2])[0])
    interp2=interp1[1:]
    if len(interp2)==0:
        i=i+1
    else:
        for j in interp2:
            index=np.where(b==j)
            for k in range(0,len(index[0])):
                if b[index[0][k],index[1][k]]>i:
                    b[index[0][k],index[1][k]]=i
        i=i+1



outletvessel=[]
connection2=[]
connection3=[]

for i in range(0,len(a)):
    interp=np.where(b==i)
    if len(interp[0])==0:
        continue
    elif len(interp[0])==1:
        outletvessel.append(interp[0][0])
    elif len(interp[0])>2:
        for j in range(0,len(interp[0])):
            connection2.append(interp[0][j])
        if len(interp[0])>3:
            for j in range(0,len(interp[0])):
                connection3.append(interp[0][j])

outletvessel=np.unique(outletvessel)
connection2=np.unique(connection2)

def in_list(item,L):
    for i in range(0,len(L)):
        if item in L[i]:
            return 1
            break
    return 0
        
all_vessels=[]
for j in outletvessel:
    vessel=[]
    k=0
    vessel.append(j)
    while k < 1000:
        i=vessel[k]
        cont=in_list(i,all_vessels)
        if cont==1:
            break
        if i in connection2 and i!=j:
            break
        if i in outletvessel and i!=j:
            break
        start_point=b[i,0]
        end_point=b[i,1]
        con1=np.where(b==start_point)[0]
        con2=np.where(b==end_point)[0]
        if len(con1)>2 or len(con2)>2:
            break
        elif len(con1)==1:
            con3=np.where(i!=con2)
            vessel.append(con2[con3[0][0]])
        elif len(con2)==1:
            con3=np.where(i!=con1)
            vessel.append(con1[con3[0][0]]) 
        else:
            con3=np.where(vessel[k-1]!=con1)
            con4=np.where(vessel[k-1]!=con2)
            if len(con3[0])==2:
                con5=np.where(i!=con1)
                vessel.append(con1[con5[0][0]])
                
            if len(con4[0])==2:
                con6=np.where(i!=con2)
                vessel.append(con2[con6[0][0]])
        k=k+1     
    all_vessels.append(vessel)     

for j in connection2:
    cont=in_list(j,all_vessels)
    if cont==1:
        continue
    else:
        vessel=[]
        k=0
        vessel.append(j)
        while k < 1000:
            i=vessel[k]
            cont=in_list(i,all_vessels)
            if cont==1:
                break
            if i in connection2 and i!=j:
                break
            if i in outletvessel and i!=j:
                break
            
            start_point=b[i,0]
            end_point=b[i,1]
            con1=np.where(b==start_point)[0]
            con2=np.where(b==end_point)[0]
            if i!=j:
                if len(con1)>2 or len(con2)>2:
                    break
                elif len(con1)==0 or len(con2)==0:
                    break
                elif len(con1)==1:
                    con3=np.where(i!=con2)
                    vessel.append(con2[con3[0][0]])
                elif len(con2)==1:
                    con3=np.where(i!=con1)
                    vessel.append(con1[con3[0][0]]) 
                else:
                    con3=np.where(vessel[k-1]!=con1)
                    con4=np.where(vessel[k-1]!=con2)
                    if len(con3[0])==2:
                        con5=np.where(i!=con1)
                        vessel.append(con1[con5[0][0]])
                        
                    if len(con4[0])==2:
                        con6=np.where(i!=con2)
                        vessel.append(con2[con6[0][0]])
                k=k+1 
            else:
                 if len(con1)>2 and len(con2)>2:
                     break
                 if len(con1)==2:
                     con3=np.where(i!=con1)
                     vessel.append(con1[con3[0][0]])
                 elif len(con2)==2:
                     con3=np.where(i!=con2)
                     vessel.append(con2[con3[0][0]]) 
                 else:
                     con3=np.where(vessel[k-1]!=con1)
                     con4=np.where(vessel[k-1]!=con2)
                     if len(con3[0])==2:
                         con5=np.where(i!=con1)
                         vessel.append(con1[con5[0][0]])
                         
                     if len(con4[0])==2:
                         con6=np.where(i!=con2)
                         vessel.append(con2[con6[0][0]])
                 k=k+1 
        all_vessels.append(vessel)          
    


length=np.zeros(len(all_vessels))
for i in range(0,len(all_vessels)):
    for j in all_vessels[i]:
        start_point=a[b[j,0],:]
        end_point=a[b[j,1],:]
        length[i]=length[i]+np.sqrt((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2+(start_point[2]-end_point[2])**2)
        
rad_a=np.zeros((len(all_vessels),2))
for i in range(0,len(all_vessels)):
    startpoin=b[all_vessels[i][0],0]
    endtpoin=b[all_vessels[i][-1],1]
    rad_a[i,0]=rad[startpoin]*rad[startpoin]*3.14/1000000
    rad_a[i,1]=rad[endtpoin]*rad[endtpoin]*3.14/1000000
    
    
rad_v=(rad_a[:,0]+rad_a[:,1])/2

inlet=np.array([[1,'Q',0,0],[2,'B',1,3],[3,'B',1,2],[4,'B',2,5],[5,'B',2,4],[6,'B',3,7],
                 [7,'B',3,6],[8,'B',4,9],[9,'B',4,8],[10,'B',5,11],
                 
                 [11,'B',5,10],[12,'B',6,13],[13,'B',6,12],[14,'B',9,15],[15,'B',9,14],[16,'B',7,17],
                 [17,'B',7,16],[18,'C',14,17],[19,'B',18,20],[20,'B',18,19],[21,'B',20,22],[22,'B',20,21],
                 [23,'B',22,24],[24,'B',20,23],[25,'B',19,26],[26,'B',19,25],[27,'C',11,26],[28,'C',12,23],
                 [29,'B',27,31],[30,'B',28,32],[31,'B',27,29],[32,'B',28,30],[33,'B',31,35],[34,'C',32,35],
                 [35,'B',31,33],[36,'B',25,37],
                 
                 [37,'B',25,36],[38,'B',37,39],[39,'B',37,38],[40,'B',38,41],[41,'B',38,40],[42,'B',41,43],
                 [43,'B',41,42],[44,'B',43,45],[45,'B',43,44],[46,'B',45,47],[47,'B',45,46],[48,'B',39,49],
                 [49,'B',39,48],[50,'B',49,51],[51,'B',49,50],[52,'B',51,53],[53,'B',51,52],[54,'B',53,55],
                 [55,'B',53,54],
                 
                 [56,'B',24,57],[57,'B',24,56],[58,'B',57,59],[59,'B',57,58],[60,'B',59,61],[61,'B',59,60],
                 [62,'B',61,63],[63,'B',61,62],[64,'B',60,65],[65,'B',60,64],[66,'B',65,67],[67,'B',65,66],
                 
                 [68,'B',30,69],[69,'B',30,68],[70,'B',69,71],[71,'B',69,70],[72,'B',71,73],[73,'B',71,72],
                 [74,'B',73,75],[75,'B',73,74],[76,'B',75,77],[77,'B',75,76],[78,'B',77,79],[79,'B',77,78],
                 [80,'B',79,81],[81,'B',79,80],[82,'B',81,83],[83,'B',81,82],[84,'B',76,85],[85,'B',76,84],
                 [86,'B',84,87],[87,'B',84,86],[88,'B',87,89],[89,'B',87,88],[90,'B',85,91],[91,'B',85,90],
                 [92,'B',91,93],[93,'B',91,92],[94,'B',93,95],[95,'B',93,94],[96,'B',72,97],[97,'B',72,96],
                 [98,'B',97,99],[99,'B',97,98],[100,'B',99,101],[101,'B',99,100],[102,'B',101,103],
                 [103,'B',101,102],[104,'B',103,105],[105,'B',103,104],[106,'B',105,107],[107,'B',105,106],
                 [108,'B',104,109],[109,'B',104,108],[110,'B',108,111],[111,'B',108,110],[112,'B',111,113],
                 [113,'B',111,112],[114,'B',109,115],[115,'B',109,114],[116,'B',115,117],[117,'B',115,116],
                 [118,'B',98,119],[119,'B',98,118],[120,'B',118,121],[121,'B',118,120],[122,'B',119,123],
                 [123,'B',119,122],[124,'B',122,125],[125,'B',122,124],[126,'B',123,127],[127,'B',123,126],
                 [128,'B',127,129],[129,'B',127,128],
                 
                 
                 [130,'B',29,133],[131,'J',130,130],[132,'B',132,132],[133,'B',29,130],[134,'B',133,135],
                 [135,'B',133,134],[136,'B',134,137],[137,'B',134,136],[138,'B',137,139],[139,'B',137,138],
                 [140,'B',136,141],[141,'B',136,140],[142,'B',141,143],[143,'B',141,142],[144,'B',143,145],
                 [145,'B',143,144],[146,'B',145,147],[147,'B',145,146],[148,'B',147,149],[149,'B',147,148],
                 [150,'B',144,151],[151,'B',144,150],[152,'B',151,153],[153,'B',151,152],[154,'B',153,155],
                 [155,'B',153,154],[156,'B',135,157],[157,'B',135,156],[158,'B',157,159],[159,'B',157,158],
                 [160,'B',158,161],[161,'B',158,160],[162,'B',161,163],[163,'B',161,162],[164,'B',163,165],
                 [165,'B',163,164],[166,'B',159,167],[167,'B',159,166],[168,'B',167,169],[169,'B',167,168],
                 [170,'B',169,171],[171,'B',169,170],[172,'B',171,173],[173,'B',171,172],[174,'B',173,175],
                 [175,'B',173,174],[176,'B',175,177],[177,'B',175,176],[178,'B',177,179],[179,'B',177,178],
                 [180,'B',168,181],[181,'B',168,180],[182,'B',181,183],[183,'B',181,182],[184,'B',182,185],
                 [185,'B',182,184],[186,'B',183,187],[187,'B',183,186],[188,'B',180,189],[189,'B',180,188],
                 [190,'B',189,191],[191,'B',189,190],[192,'B',191,193],[193,'B',191,192],[194,'B',193,195],
                 [195,'B',193,194],[196,'B',195,197],[197,'B',195,196],[198,'B',192,199],[199,'B',192,198],
                 [200,'B',199,201],[201,'B',200,199],[202,'B',201,203],[203,'B',201,202],[204,'B',203,205],
                 [205,'B',203,204],[206,'B',205,207],[207,'B',205,206],[208,'B',207,209],[209,'B',207,208],
                 [210,'B',209,211],[211,'B',209,210],
                 
                 
                 [212,'B',33,213],[213,'B',33,212],[214,'B',213,215],[215,'B',213,214],[216,'B',214,217],
                 [217,'B',214,216],[218,'B',217,219],[219,'B',217,218],[220,'B',215,221],[221,'B',215,220],
                 [222,'B',221,223],[223,'B',221,222],[224,'B',222,225],[225,'B',222,224],[226,'B',225,227],
                 [227,'B',225,226],[228,'B',223,229],[229,'B',223,228],[230,'B',229,231],[231,'B',229,230],
                 [232,'B',231,233],[233,'B',231,232],[234,'B',233,235],[235,'B',233,234],[236,'B',235,237],
                 [237,'B',235,236],[238,'B',234,239],[239,'B',234,238],[240,'B',239,241],[241,'B',239,240],
                 [242,'B',241,243],[243,'B',241,242],[244,'B',230,245],[245,'B',230,244],[246,'B',245,247],
                 [247,'B',245,246],
                 
                 [248,'B',34,249],[249,'B',34,248],[250,'B',248,251],[251,'B',248,250],[252,'B',251,253],
                 [253,'B',251,252],[254,'B',249,255],[255,'B',249,254],[256,'B',255,257],[257,'B',255,256],
                 [258,'B',257,259],[259,'B',257,258],[260,'B',259,261],[261,'B',259,260],[262,'B',256,263],
                 [263,'B',256,262],[264,'B',263,265],[265,'B',263,264],[266,'B',265,267],[267,'B',265,266],
                 [268,'B',267,269],[269,'B',267,268],[270,'B',264,271],[271,'B',264,270],[272,'B',271,273],
                 [273,'B',271,272],[274,'B',273,275],[275,'B',273,274],[276,'B',272,277],[277,'B',272,276],
                 [278,'B',277,279],[279,'B',277,278],[280,'B',279,281],[281,'B',279,280]
                 
                 
                 
])


outlet=np.array([[1,'B',2,3],[2,'B',4,5],[3,'B',6,7],[4,'B',8,9],[5,'B',10,11],[6,'B',12,13],
                 [7,'B',16,17],[8,'W',0,0],[9,'B',14,15],[10,'W',0,0],
                 
                 [11,'C',26,27],[12,'C',23,28],[13,'W',0,0],[14,'C',17,18],[15,'W',0,0],[16,'W',0,0],
                 [17,'C',14,18],[18,'B',19,20],[19,'B',25,26],[20,'B',21,22],[21,'W',0,0],[22,'B',23,24],
                 [23,'C',12,28],[24,'B',56,57],[25,'B',36,37],[26,'C',11,27],[27,'B',29,31],[28,'B',30,32],
                 [29,'B',130,133],[30,'B',68,69],[31,'B',33,35],[32,'C',35,34],[33,'B',212,213],[34,'B',248,249],
                 [35,'C',32,34],[36,'W',0,0],
                 
                 [37,'B',38,39],[38,'B',40,41],[39,'B',48,49],[40,'W',0,0],[41,'B',42,43],[42,'W',0,0],
                 [43,'B',44,45],[44,'W',0,0],[45,'B',46,47],[46,'W',0,0],[47,'W',0,0],[48,'W',0,0],
                 [49,'B',50,51],[50,'W',0,0],[51,'B',52,53],[52,'W',0,0],[53,'B',54,55],[54,'W',0,0],
                 [55,'W',0,0],
                 
                 [56,'W',0,0],[57,'B',58,59],[58,'W',0,0],[59,'B',60,61],[60,'B',64,65],[61,'B',62,63],
                 [62,'W',0,0],[63,'W',0,0],[64,'W',0,0],[65,'B',66,67],[66,'W',0,0],[67,'W',0,0],
                 
                 [68,'W',0,0],[69,'B',70,71],[70,'W',0,0],[71,'B',72,73],[72,'B',96,97],[73,'B',74,75],
                 [74,'W',0,0],[75,'B',76,77],[76,'B',84,85],[77,'B',78,79],[78,'W',0,0],[79,'B',80,81],
                 [80,'W',0,0],[81,'B',82,83],[82,'W',0,0],[83,'W',0,0],[84,'B',86,87],[85,'B',90,91],
                 [86,'W',0,0],[87,'B',88,89],[88,'W',0,0],[89,'W',0,0],[90,'W',0,0],[91,'B',92,93],
                 [92,'W',0,0],[93,'B',94,95],[94,'W',0,0],[95,'W',0,0],[96,'W',0,0],[97,'B',98,99],
                 [98,'B',118,119],[99,'B',100,101],[100,'W',0,0],[101,'B',102,103],[102,'W',0,0],
                 [103,'B',104,105],[104,'B',108,109],[105,'B',106,107],[106,'W',0,0],[107,'W',0,0],
                 [108,'B',110,111],[109,'B',114,115],[110,'W',0,0],[111,'B',112,113],[112,'W',0,0],
                 [113,'W',0,0],[114,'W',0,0],[115,'B',116,117],[116,'W',0,0],[117,'W',0,0],
                 [118,'B',120,121],[119,'B',122,123],[120,'W',0,0],[121,'W',0,0],[122,'B',124,125],
                 [123,'B',126,127],[124,'W',0,0],[125,'W',0,0],[126,'W',0,0],[127,'B',128,129],
                 [128,'W',0,0],[129,'W',0,0],
                 
                 
                 [130,'J',131,131],[131,'J',132,132],[132,'W',0,0],[133,'B',134,135],[134,'B',136,137],
                 [135,'B',156,157],[136,'B',140,141],[137,'B',138,139],[138,'W',0,0],[139,'W',0,0],
                 [140,'W',0,0],[141,'B',142,143],[142,'W',0,0],[143,'B',144,145],[144,'B',150,151],
                 [145,'B',146,147],[146,'W',0,0],[147,'B',148,149],[148,'W',0,0],[149,'W',0,0],
                 [150,'W',0,0],[151,'B',152,153],[152,'W',0,0],[153,'B',154,155],[154,'W',0,0],
                 [155,'W',0,0],[156,'W',0,0],[157,'B',158,159],[158,'B',160,161],[159,'B',166,167],
                 [160,'W',0,0],[161,'B',162,163],[162,'W',0,0],[163,'B',164,165],[164,'W',0,0],
                 [165,'W',0,0],[166,'W',0,0],[167,'B',168,169],[168,'B',180,181],[169,'B',170,171],
                 [170,'W',0,0],[171,'B',172,173],[172,'W',0,0],[173,'B',174,175],[174,'W',0,0],
                 [175,'B',176,177],[176,'W',0,0],[177,'B',178,179],[178,'W',0,0],[179,'W',0,0],
                 [180,'B',188,189],[181,'B',182,183],[182,'B',184,185],[183,'B',186,187],[184,'W',0,0],
                 [185,'W',0,0],[186,'W',0,0],[187,'W',0,0],[188,'W',0,0],[189,'B',190,191],
                 [190,'W',0,0],[191,'B',192,193],[192,'B',198,199],[193,'B',194,195],[194,'W',0,0],
                 [195,'B',196,197],[196,'W',0,0],[197,'W',0,0],[198,'W',0,0],[199,'B',200,201],
                 [200,'W',0,0],[201,'B',202,203],[202,'W',0,0],[203,'B',204,205],[204,'W',0,0],
                 [205,'B',206,207],[206,'W',0,0],[207,'B',208,209],[208,'W',0,0],[209,'B',210,211],
                 [210,'W',0,0],[211,'W',0,0],
                 
                 
                 [212,'W',0,0],[213,'B',214,215],[214,'B',216,217],[215,'B',220,221],[216,'W',0,0],
                 [217,'B',218,219],[218,'W',0,0],[219,'W',0,0],[220,'W',0,0],[221,'B',222,223],
                 [222,'B',224,225],[223,'B',228,229],[224,'W',0,0],[225,'B',226,227],[226,'W',0,0],
                 [227,'W',0,0],[228,'W',0,0],[229,'B',230,231],[230,'B',244,245],[231,'B',232,233],
                 [232,'W',0,0],[233,'B',234,235],[234,'B',238,239],[235,'B',236,237],[236,'W',0,0],
                 [237,'W',0,0],[238,'W',0,0],[239,'B',240,241],[240,'W',0,0],[241,'B',242,243],
                 [242,'W',0,0],[243,'W',0,0],[244,'W',0,0],[245,'B',246,247],[246,'W',0,0],
                 [247,'W',0,0],
                 
                 [248,'B',250,251],[249,'B',254,255],[250,'W',0,0],[251,'B',252,253],[252,'W',0,0],
                 [253,'W',0,0],[254,'W',0,0],[255,'B',256,257],[256,'B',262,263],[257,'B',258,259],
                 [258,'W',0,0],[259,'B',260,261],[260,'W',0,0],[261,'W',0,0],[262,'W',0,0],
                 [263,'B',264,265],[264,'B',270,271],[265,'B',266,267],[266,'W',0,0],[267,'B',268,269],
                 [268,'W',0,0],[269,'W',0,0],[270,'W',0,0],[271,'B',272,273],[272,'B',276,277],
                 [273,'B',274,275],[274,'W',0,0],[275,'W',0,0],[276,'W',0,0],[277,'B',278,279],
                 [278,'W',0,0],[279,'B',280,281],[280,'W',0,0],[281,'W',0,0]
                                                                         ])
MCAL=np.arange(68,130,1)
MCAR=np.arange(130,212,1)
ACAR=np.arange(212,248,1)
ACAL=np.arange(248,282,1)
PCAR=np.arange(37,56,1)
PCAL=np.arange(56,68,1)

np.savetxt('CoW_0_ACAL.csv',ACAL,delimiter=',')
np.savetxt('CoW_0_ACAR.csv',ACAR,delimiter=',')
np.savetxt('CoW_0_MCAL.csv',MCAL,delimiter=',')
np.savetxt('CoW_0_MCAR.csv',MCAR,delimiter=',')
np.savetxt('CoW_0_PCAL.csv',PCAL,delimiter=',')
np.savetxt('CoW_0_PCAR.csv',PCAR,delimiter=',')

for patient in range(0,1):
    
    
    number=(patient+1)*10
    i=patient*10
    
    
    T=1 #s
    dT=5e-6 #s 
    
    lowertime=40 #min
    uppertime=lowertime+T
    maxtime=uppertime+(T/2)
    Timenumber=maxtime/dT
    PI=3.142
    
    Age=np.linspace(30, 80, num=(number-i))
    np.savetxt('Age.csv',Age,delimiter=',')
    
    Psys=17300 #Pa
    Pdia=10100 #Pa
    SV=70*1e-6 #m/s
    T=1
    
    L=np.array([40,20,34,39,208,177,34,156,34,177,length[8],length[10],177,length[113],422,422,
                length[100],length[249],
                
                length[233],length[224],length[97],length[225],length[234],length[230],
                length[226],length[184],length[183],length[235],length[203],length[157],
                length[166],length[240],length[168],length[248],length[167],
                
                length[99],length[227],length[228],length[236],length[101],length[229],
                length[112],length[238],length[108],length[222],length[95],length[102],
                length[105],length[237],length[111],length[219],length[93],length[220],
                length[98],length[106],
                
                length[103],length[231],length[104],length[232],length[239],length[223],
                length[96],length[107],length[109],length[221],length[94],length[110],
                
                length[21],length[158],length[33],length[171],length[200],length[193],
                length[56],length[199],length[246],length[207],length[63],length[164],
                length[25],length[165],length[48],length[75],length[170],length[173],
                length[31],length[141],length[43],length[6],length[36],length[174],
                length[62],length[153],length[18],length[29],length[58],length[201],
                length[242],length[217],length[84],length[178],length[39],length[179],
                length[190],length[180],length[40],length[74],length[218],length[148],
                length[90],length[151],length[65],length[16],length[14],length[134],
                length[0],length[27],
                
                length[59]/3,length[59]/3,length[59]/3,length[202],length[245],length[194],
                length[191],length[185],length[44],length[81],length[51],length[192],
                length[82],length[216],length[186],length[181],length[41],length[143],
                length[9],length[72],length[46],length[187],length[91],length[152],
                length[17],length[92],length[52],length[193],length[215],length[210],
                length[80],length[160],length[77],length[159],length[22],length[79],
                length[68],length[176],length[247],length[208],length[64],length[149],
                length[15],length[150],length[88],length[211],length[67],length[144],
                length[11],length[53],length[189],length[145],length[169],length[12],
                length[32],length[57],length[28],length[47],length[188],length[71],
                length[212],length[209],length[177],length[38],length[137],length[89],
                length[3],length[67],length[147],length[13],length[146],length[87],
                length[156],length[20],length[155],length[50],length[182],length[208],
                length[144],length[53],length[11],
                
                length[133],length[266],length[265],length[262],length[121],length[264],
                length[129],length[131],length[127],length[263],length[257],length[259],
                length[120],length[250],length[115],length[132],length[125],length[260],
                length[254],length[251],length[116],length[252],length[261],length[258],
                length[124],length[122],length[126],length[255],length[119],length[256],
                length[130],length[123],
                
                length[244],length[205],length[26],length[251],length[85],length[73],
                length[61],length[206],length[196],length[163],length[24],length[135],
                length[1],length[86],length[55],length[197],length[138],length[142],
                length[7],length[140],length[66],length[5],length[4],length[139],
                length[204],length[195],length[54],length[30],length[60],length[161],
                length[23],length[162],length[83],length[45]
                ])/1000
    
    E=np.array([0.4,0.8,1.6])*1e6
    
    
    
    
    while i < 1:
        
        
        Agecase=Age[i-patient*10:i-patient*10+1]
        name='Age_'+str(i)+'.npy'
        np.save(name,Agecase)
        Lcase=L
        
        
        
        Ao=np.array([ata.ascending(Agecase)[0],ata.arch(Agecase[0])[0][0],ata.BCT(Agecase),ata.arch(Agecase[0])[0][0],
                     ata.CCA(Agecase),ata.CCA(Agecase),ata.LS(Agecase),ata.descending(Agecase)[0],ata.LS(Agecase),
                     ata.ICAL(Agecase)[0],rad_v[8],rad_v[10],ata.ICAR(Agecase)[0],rad_v[113],PI*(0.403)*(0.403)/(100*100),
                     PI*(0.403)*(0.403)/(100*100),rad_v[100],rad_v[249],
                    
                    rad_v[233],rad_v[224],rad_v[97],rad_v[225],rad_v[234],rad_v[230],
                    rad_v[226],rad_v[184],rad_v[183],rad_v[235],rad_v[203],rad_v[157],
                    rad_v[166],rad_v[240],rad_v[168],rad_v[248],rad_v[167],
                    
                    rad_v[99],rad_v[227],rad_v[228],rad_v[236],rad_v[101],rad_v[229],
                    rad_v[112],rad_v[238],rad_v[108],rad_v[222],rad_v[95],rad_v[102],
                    rad_v[105],rad_v[237],rad_v[111],rad_v[219],rad_v[93],rad_v[220],
                    rad_v[98],rad_v[106],
                    
                    rad_v[103],rad_v[231],rad_v[104],rad_v[232],rad_v[239],rad_v[223],
                    rad_v[96],rad_v[107],rad_v[109],rad_v[221],rad_v[94],rad_v[110],
                    
                    rad_v[21],rad_v[158],rad_v[33],rad_v[171],rad_v[200],rad_v[193],
                    rad_v[56],rad_v[199],rad_v[246],rad_v[207],rad_v[63],rad_v[164],
                    rad_v[25],rad_v[165],rad_v[48],rad_v[75],rad_v[170],rad_v[173],
                    rad_v[31],rad_v[141],rad_v[43],rad_v[6],rad_v[36],rad_v[174],
                    rad_v[62],rad_v[153],rad_v[18],rad_v[29],rad_v[58],rad_v[201],
                    rad_v[242],rad_v[217],rad_v[84],rad_v[178],rad_v[39],rad_v[179],
                    rad_v[190],rad_v[180],rad_v[40],rad_v[74],rad_v[218],rad_v[148],
                    rad_v[90],rad_v[151],rad_v[65],rad_v[16],rad_v[14],rad_v[134],
                    rad_v[0],rad_v[27],
                    
                    rad_v[59],rad_v[59],rad_v[59],rad_v[202],rad_v[245],rad_v[194],
                    rad_v[191],rad_v[185],rad_v[44],rad_v[81],rad_v[51],rad_v[192],
                    rad_v[82],rad_v[216],rad_v[186],rad_v[181],rad_v[41],rad_v[143],
                    rad_v[9],rad_v[72],rad_v[46],rad_v[187],rad_v[91],rad_v[152],
                    rad_v[17],rad_v[92],rad_v[52],rad_v[193],rad_v[215],rad_v[210],
                    rad_v[80],rad_v[160],rad_v[77],rad_v[159],rad_v[22],rad_v[79],
                    rad_v[68],rad_v[176],rad_v[247],rad_v[208],rad_v[64],rad_v[149],
                    rad_v[15],rad_v[150],rad_v[88],rad_v[211],rad_v[67],rad_v[144],
                    rad_v[11],rad_v[53],rad_v[189],rad_v[145],rad_v[169],rad_v[12],
                    rad_v[32],rad_v[57],rad_v[28],rad_v[47],rad_v[188],rad_v[71],
                    rad_v[212],rad_v[209],rad_v[177],rad_v[38],rad_v[137],rad_v[89],
                    rad_v[3],rad_v[67],rad_v[147],rad_v[13],rad_v[146],rad_v[87],
                    rad_v[156],rad_v[20],rad_v[155],rad_v[50],rad_v[182],rad_v[208],
                    rad_v[144],rad_v[53],rad_v[11],
                    
                    rad_v[133],rad_v[266],rad_v[265],rad_v[262],rad_v[121],rad_v[264],
                    rad_v[129],rad_v[131],rad_v[127],rad_v[263],rad_v[257],rad_v[259],
                    rad_v[120],rad_v[250],rad_v[115],rad_v[132],rad_v[125],rad_v[260],
                    rad_v[254],rad_v[251],rad_v[116],rad_v[252],rad_v[261],rad_v[258],
                    rad_v[124],rad_v[122],rad_v[126],rad_v[255],rad_v[119],rad_v[256],
                    rad_v[130],rad_v[123],
                    
                    rad_v[244],rad_v[205],rad_v[26],rad_v[251],rad_v[85],rad_v[73],
                    rad_v[61],rad_v[206],rad_v[196],rad_v[163],rad_v[24],rad_v[135],
                    rad_v[1],rad_v[86],rad_v[55],rad_v[197],rad_v[138],rad_v[142],
                    rad_v[7],rad_v[140],rad_v[66],rad_v[5],rad_v[4],rad_v[139],
                    rad_v[204],rad_v[195],rad_v[54],rad_v[30],rad_v[60],rad_v[161],
                    rad_v[23],rad_v[162],rad_v[83],rad_v[45]
                    ])
        
        
        Aocase=Ao
         
        a = 0.2802 #
        b = -0.5053*1000 #m-1
        c = 0.1324 #
        d = -0.01114*1000 #m-1
        r0=np.zeros(len(Lcase))
        h=np.zeros(len(Lcase))
        for j in range(0,len(Lcase)):
            r0[j]=np.sqrt(Aocase[j]/PI)
            h[j]=r0[j]*(a*np.exp(b*r0[j])+c*np.exp(d*r0[j]))
            
        a0,an1,an2,an3,an4,bn1,bn2,bn3,bn4,HR=fr.maxminvalues(0,0,series)
        
        HR=1/HR
        
        Rtot=((1/3)*Psys+(2/3)*Pdia)/(SV*HR)
        Ctot = T/Rtot
    
        C1d=np.zeros(len(h))
        for j in range(0,len(h)):
            if j<8:
                C1d[j]=(3*Aocase[j]*np.sqrt(Aocase[j])*Lcase[j])/(2*np.sqrt(PI)*E[0]*h[j])
            elif j<14 and  j>=8:
                C1d[j]=(3*Aocase[j]*np.sqrt(Aocase[j])*Lcase[j])/(2*np.sqrt(PI)*E[1]*h[j])
            else:
                C1d[j]=(3*Aocase[j]*np.sqrt(Aocase[j])*Lcase[j])/(2*np.sqrt(PI)*E[2]*h[j])
            
        Rt=np.zeros(len(h))
        R1=np.zeros(len(h))
        C=np.zeros(len(h))
        beta=np.zeros(len(h))
        Cl0=np.zeros(len(h))
        r0sum=0
        
        for j in range(0,len(h)):
            if outlet[j,1]=='W':
                r0sum=r0sum+r0[j]
                
        for j in range(0,len(h)):
            Rt[j]=(Rtot/0.2)*(r0sum/r0[j])
            C[j]=np.abs((Ctot-np.sum(C1d))*(Rtot/Rt[j]))
            if j<8:
                beta[j]=(4/3)*((np.sqrt(np.pi)*E[0]*h[j])/(Ao[j]))
            elif j<14:
                beta[j]=(4/3)*((np.sqrt(np.pi)*E[1]*h[j])/(Ao[j]))
            else:
                beta[j]=(4/3)*((np.sqrt(np.pi)*E[2]*h[j])/(Ao[j]))
            Cl0[j]=np.sqrt(0.5*beta[j]/1060)*np.power(Ao[j],0.25)
            R1[j]=1060*Cl0[j]/Ao[j]

        
        an=np.zeros(4)
        bn=np.zeros(4)
        x=np.linspace(0,maxtime, num=int(Timenumber)+1)
        
        a01=a0
        an[0]=an1
        an[1]=an2
        an[2]=an3
        an[3]=an4
        bn[0]=bn1
        bn[1]=bn2
        bn[2]=bn3
        bn[3]=bn4
        
            
    
        for j in [0]:        
            
       
                
            name='CoW_'+str(i)+'_'+str(j)+'.in'
            f = open(name,"w")
            f.write("10 parameter list\n")
            f.write("0		EQTYPE\n")
            f.write("{0} 	DT\n".format(dT))
            f.write("{0} 	NSTEPS\n".format(int(Timenumber)))
            f.write("{0} 	IOSTEP\n".format(int(Timenumber)))
            f.write("1E3	    HISSTEP\n")
            f.write("2		INTTYPE\n")
            f.write("1060		Rho (Kg/m3)\n")
            f.write("1 	Alpha (velocity profile param in the momentum eq) (for Alpha=1 no wall viscous effects considered)\n")
            f.write("4.0e-3		Viscosity (Pa*s)\n")
            f.write("3.142e+00 	 PI\n")
            f.write("Mesh  -- expansion order -- quadrature order Ndomains = {0}\n".format(len(L)))
            for k in range(0,len(Lcase)):
                if k<8:
                    f.write('''1       nel  Eh Area Domain {3}
0.0  {0}  3  3 # x_lower x_upper L q
Ao = {1}
Eh = {4}*{2}\n'''.format(Lcase[k],Aocase[k],h[k],(k+1),E[0]))
                elif k<14 and  k>=8:
                    f.write('''1       nel  Eh Area Domain {3}
0.0  {0}  3  3 # x_lower x_upper L q
Ao = {1}
Eh = {4}*{2}\n'''.format(Lcase[k],Aocase[k],h[k],(k+1),E[1]))
                else :
                    f.write('''1       nel  Eh Area Domain {3}
0.0  {0}  3  3 # x_lower x_upper L q
Ao = {1}
Eh = {4}*{2}\n'''.format(Lcase[k],Aocase[k],h[k],(k+1),E[2]))
    
            f.write("Boundary conditions\n")
            
            for k in range(0,len(L)):
                if inlet[k,1]=='Q':
                        f.write('''q  0 	 # Domain {0}
    q = ({4:.2e}/2)+({5:.2e}*cos(6.28*t))+({6:.2e}*cos(6.28*2*t))+({7:.2e}*cos(6.28*3*t))+({8:.2e}*cos(6.28*4*t))+({9:.2e}*sin(6.28*t))+({10:.2e}*sin(6.28*2*t))+({11:.2e}*sin(6.28*3*t))+({12:.2e}*sin(6.28*4*t))
q  0
    q = ({4:.2e}/2)+({5:.2e}*cos(6.28*t))+({6:.2e}*cos(6.28*2*t))+({7:.2e}*cos(6.28*3*t))+({8:.2e}*cos(6.28*4*t))+({9:.2e}*sin(6.28*t))+({10:.2e}*sin(6.28*2*t))+({11:.2e}*sin(6.28*3*t))+({12:.2e}*sin(6.28*4*t))
{1}  {2}  {3}
{1}  {2}  {3}\n'''.format(k+1,outlet[k,1],int(outlet[k,2]),int(outlet[k,3]),a01,an[0],an[1],an[2],an[3],bn[0],bn[1],bn[2],bn[3]))
                elif outlet[k,1]=='W':
                    f.write('''{1}  {2}  {3} 	 # Domain {0}
{1}  {2}  {3}\n'''.format(k+1,inlet[k,1],int(inlet[k,2]),int(inlet[k,3])))
                    f.write('''W  {:.2e}
W  {:.2e}\n'''.format(C[k],Rt[k]))
                else:
                    f.write('''{1}  {2}  {3} 	 # Domain {0}
{1}  {2}  {3}
{4}  {5}  {6}
{4}  {5}  {6}\n'''.format(k+1,inlet[k,1],int(inlet[k,2]),int(inlet[k,3]),outlet[k,1],int(outlet[k,2]),int(outlet[k,3])))

     
            f.write("Initial condition\n")
            for k in range(0,len(L)):
                f.write('''a = Ao
u = 0.0\n''')
    
            f.write('''History Pts
8 #Number of Domains with history points
1 5  #Npts Domain id x[1], x[2], x[3]
0.1
1 6  #Npts Domain id x[1], x[2], x[3]
0.1\n''')    

            f.write('''1 {0}  #Npts Domain id x[1], x[2], x[3]
0\n'''.format(int(ACAL[0])))
            f.write('''1 {0}  #Npts Domain id x[1], x[2], x[3]
0\n'''.format(int(ACAR[0])))
            f.write('''1 {0}  #Npts Domain id x[1], x[2], x[3]
0\n'''.format(int(MCAL[0])))
            f.write('''1 {0}  #Npts Domain id x[1], x[2], x[3]
0\n'''.format(int(MCAR[0])))
            f.write('''1 {0}  #Npts Domain id x[1], x[2], x[3]
0\n'''.format(int(PCAL[0])))
            f.write('''1 {0}  #Npts Domain id x[1], x[2], x[3]
0\n'''.format(int(PCAR[0])))

            f.close()
        i=i+1