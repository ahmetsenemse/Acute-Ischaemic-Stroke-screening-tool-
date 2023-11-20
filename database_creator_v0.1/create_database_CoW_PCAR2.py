import numpy as np
import artery_area as ata
from sklearn.utils import shuffle


inlet=np.array([[1,'Q',0,0],[2,'B',1,3],[3,'B',1,2],[4,'B',2,5],[5,'B',2,4],[6,'B',3,7],
                 [7,'B',3,6],[8,'B',4,9],[9,'B',4,8],[10,'B',5,11],[11,'B',5,10],[12,'B',6,13],
                 [13,'B',6,12],[14,'B',9,15],[15,'B',9,14],[16,'B',7,17],[17,'B',7,16],[18,'C',14,17],
                 
                 [19,'B',18,32],[20,'B',19,21],[21,'B',19,20],[22,'B',21,23],[23,'B',21,22],[24,'B',22,25],
                 [25,'B',22,24],[26,'B',24,27],[27,'B',24,26],[28,'B',26,29],[29,'B',26,28],[30,'B',28,31],
                 [31,'B',28,30],
                 
                 [32,'B',18,19],[33,'B',32,34],[34,'B',32,33],[35,'B',34,36],[36,'B',34,35],[37,'B',35,38],
                 [38,'B',35,37],[39,'B',37,40],[40,'B',37,39],[41,'B',39,42],[42,'B',39,41],[43,'B',41,44],
                 [44,'B',41,43],
                 
                 [45,'C',11,20],[46,'B',12,130],[47,'B',45,50],[48,'B',46,68],[49,'B',47,86],
                 
                 [50,'B',45,47],[51,'B',50,58],[52,'B',51,53],[53,'B',51,52],[54,'B',52,55],[55,'B',52,54],
                 [56,'B',53,57],[57,'B',53,56],[58,'B',50,51],[59,'B',58,60],[60,'B',58,59],[61,'B',59,63],
                 [62,'B',60,52],[63,'B',59,61],[64,'J',61,61],[65,'B',60,62],[66,'B',62,67],[67,'B',62,66],
                 
                 [68,'B',46,48],[69,'B',68,76],[70,'B',69,71],[71,'B',69,70],[72,'B',70,73],[73,'B',70,72],
                 [74,'B',71,75],[75,'B',71,74],[76,'B',68,69],[77,'B',76,78],[78,'B',76,77],[79,'B',77,81],
                 [80,'B',78,83],[81,'B',77,79],[82,'J',79,79],[83,'B',78,80],[84,'B',80,85],[85,'B',80,84],
                 
                 [86,'B',47,49],[87,'B',86,107],[88,'B',87,94],[89,'J',88,88],[90,'B',89,93],[91,'B',90,92],
                 [92,'B',90,91],[93,'B',89,90],[94,'B',87,88],[95,'B',94,106],[96,'B',95,105],[97,'B',96,104],
                 [98,'B',97,103],[99,'B',98,102],[100,'J',99,101],[101,'B',99,100],[102,'B',98,99],[103,'B',97,98],
                 [104,'B',96,97],[105,'B',95,96],[106,'J',94,95],[107,'B',86,87],
                 
                 [108,'C',48,49],[109,'B',108,129],[110,'B',109,116],[111,'J',110,110],[112,'B',111,115],[113,'B',112,114],
                 [114,'B',112,113],[115,'B',111,112],[116,'B',109,110],[117,'B',116,128],[118,'B',117,127],[119,'B',118,126],
                 [120,'B',119,125],[121,'B',120,124],[122,'J',121,123],[123,'B',121,122],[124,'B',120,121],[125,'B',119,120],
                 [126,'B',118,119],[127,'B',117,118],[128,'J',116,117],[129,'B',108,109],[130,'B',12,46]
])


outlet=np.array([[1,'B',2,3],[2,'B',4,5],[3,'B',6,7],[4,'B',8,9],[5,'B',10,11],[6,'B',12,13],
                 [7,'B',16,17],[8,'W',0,0],[9,'B',14,15],[10,'W',0,0],[11,'C',20,45],[12,'B',46,130],
                 [13,'W',0,0],[14,'C',17,18],[15,'W',0,0],[16,'W',0,0],[17,'C',14,18],[18,'B',19,32],
                 
                 [19,'B',20,21],[20,'C',11,45],[21,'B',22,23],[22,'B',24,25],[23,'W',0,0],[24,'B',26,27],
                 [25,'W',0,0],[26,'B',28,29],[27,'W',0,0],[28,'B',30,31],[29,'W',0,0],[30,'W',0,0],
                 [31,'W',0,0],
                 
                 [32,'B',33,34],[33,'T',1,1],[34,'B',35,36],[35,'B',37,38],[36,'W',0,0],[37,'B',39,40],
                 [38,'W',0,0],[39,'B',41,42],[40,'W',0,0],[41,'B',43,44],[42,'W',0,0],[43,'W',0,0],
                 [44,'W',0,0],
                 
                 [45,'B',47,50],[46,'B',48,68],[47,'B',49,86],[48,'C',49,108],[49,'C',48,108],
                 
                 [50,'B',51,58],[51,'B',52,53],[52,'B',54,55],[53,'B',56,57],[54,'W',0,0],[55,'W',0,0],
                 [56,'W',0,0],[57,'W',0,0],[58,'B',59,60],[59,'B',61,63],[60,'B',62,65],[61,'J',64,64],
                 [62,'B',66,67],[63,'W',0,0],[64,'W',0,0],[65,'W',0,0],[66,'W',0,0],[67,'W',0,0],
                 
                 [68,'B',69,76],[69,'B',70,71],[70,'B',72,73],[71,'B',74,75],[72,'W',0,0],[73,'W',0,0],
                 [74,'W',0,0],[75,'W',0,0],[76,'B',77,78],[77,'B',79,81],[78,'B',80,83],[79,'J',82,82],
                 [80,'B',84,85],[81,'W',0,0],[82,'W',0,0],[83,'W',0,0],[84,'W',0,0],[85,'W',0,0],
                 
                 [86,'B',87,107],[87,'B',88,94],[88,'J',89,89],[89,'B',90,93],[90,'B',91,92],[91,'W',0,0],
                 [92,'W',0,0],[93,'W',0,0],[94,'B',95,106],[95,'B',96,105],[96,'B',97,104],[97,'B',98,103],
                 [98,'B',99,102],[99,'B',100,101],[100,'W',0,0],[101,'W',0,0],[102,'W',0,0],[103,'W',0,0],
                 [104,'W',0,0],[105,'W',0,0],[106,'W',0,0],[107,'W',0,0],
                 
                 [108,'B',109,129],[109,'B',110,116],[110,'J',111,111],[111,'B',112,115],[112,'B',113,114],[113,'W',0,0],
                 [114,'W',0,0],[115,'W',0,0],[116,'B',117,128],[117,'B',118,127],[118,'B',119,126],[119,'B',120,125],
                 [120,'B',121,124],[121,'B',122,123],[122,'W',0,0],[123,'W',0,0],[124,'W',0,0],[125,'W',0,0],
                 [126,'W',0,0],[127,'W',0,0],[128,'W',0,0],[129,'W',0,0],[130,'T',1,1]
                                                                        
                                                                        
                 
                ])


MCAL=np.array([50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
MCAR=np.array([68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85])
ACAL=np.array([86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107])
ACAR=np.array([108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129])
PCAL=np.array([21,22,23,24,25,26,27,28,29,30,31])
PCAR=np.array([34,35,36,37,38,39,40,41,42,43,44])


inflow_random_chose=np.unique(np.random.choice(range(3), [50000,6]), axis=0)
inflow_parameters = shuffle(inflow_random_chose, random_state=0)


def parameters(patient,Agecase):    
    if Agecase==25:
        if inflow_parameters[patient,0]==0:
            HR=73-9.1
        elif inflow_parameters[patient,0]==1:
            HR=73
        elif inflow_parameters[patient,0]==2:
            HR=73+9.1
            
        if inflow_parameters[patient,1]==0:
            SV=66.8-13.1
        elif inflow_parameters[patient,1]==1:
            SV=66.8
        elif inflow_parameters[patient,1]==2:
            SV=66.8+13.1

        if inflow_parameters[patient,2]==0:
            LVET=283-23
        elif inflow_parameters[patient,2]==1:
            LVET=283
        elif inflow_parameters[patient,2]==2:
            LVET=283+23                

        if inflow_parameters[patient,3]==0:
            PFT=79.9-0.4
        elif inflow_parameters[patient,3]==1:
            PFT=79.9
        elif inflow_parameters[patient,3]==2:
            PFT=79.9+0.4                

        if inflow_parameters[patient,4]==0:
            RFV=0.7-0
        elif inflow_parameters[patient,4]==1:
            RFV=0.7
        elif inflow_parameters[patient,4]==2:
            RFV=0.7+0  

        if inflow_parameters[patient,5]==0:
            MAV=585-130
        elif inflow_parameters[patient,5]==1:
            MAV=585
        elif inflow_parameters[patient,5]==2:
            MAV=585+130

    elif Agecase==35:
        if inflow_parameters[patient,0]==0:
            HR=76.3-9.1
        elif inflow_parameters[patient,0]==1:
            HR=76.3
        elif inflow_parameters[patient,0]==2:
            HR=76.3+9.1
            
        if inflow_parameters[patient,1]==0:
            SV=64.1-12.5
        elif inflow_parameters[patient,1]==1:
            SV=64.1
        elif inflow_parameters[patient,1]==2:
            SV=64.8+12.5

        if inflow_parameters[patient,2]==0:
            LVET=284-23
        elif inflow_parameters[patient,2]==1:
            LVET=284
        elif inflow_parameters[patient,2]==2:
            LVET=284+23                

        if inflow_parameters[patient,3]==0:
            PFT=80-0
        elif inflow_parameters[patient,3]==1:
            PFT=80
        elif inflow_parameters[patient,3]==2:
            PFT=80+0                

        if inflow_parameters[patient,4]==0:
            RFV=0.7-0
        elif inflow_parameters[patient,4]==1:
            RFV=0.7
        elif inflow_parameters[patient,4]==2:
            RFV=0.7+0  

        if inflow_parameters[patient,5]==0:
            MAV=572-132
        elif inflow_parameters[patient,5]==1:
            MAV=572
        elif inflow_parameters[patient,5]==2:
            MAV=572+132

    elif Agecase==45:
        if inflow_parameters[patient,0]==0:
            HR=77-9
        elif inflow_parameters[patient,0]==1:
            HR=77
        elif inflow_parameters[patient,0]==2:
            HR=77+9
            
        if inflow_parameters[patient,1]==0:
            SV=61.3-11.6
        elif inflow_parameters[patient,1]==1:
            SV=61.3
        elif inflow_parameters[patient,1]==2:
            SV=61.8+11.6

        if inflow_parameters[patient,2]==0:
            LVET=283-23
        elif inflow_parameters[patient,2]==1:
            LVET=283
        elif inflow_parameters[patient,2]==2:
            LVET=283+23                

        if inflow_parameters[patient,3]==0:
            PFT=80-0
        elif inflow_parameters[patient,3]==1:
            PFT=80
        elif inflow_parameters[patient,3]==2:
            PFT=80+0                

        if inflow_parameters[patient,4]==0:
            RFV=0.7-0
        elif inflow_parameters[patient,4]==1:
            RFV=0.7
        elif inflow_parameters[patient,4]==2:
            RFV=0.7+0  

        if inflow_parameters[patient,5]==0:
            MAV=573-126
        elif inflow_parameters[patient,5]==1:
            MAV=573
        elif inflow_parameters[patient,5]==2:
            MAV=573+126

    elif Agecase==55:
        if inflow_parameters[patient,0]==0:
            HR=77-9.1
        elif inflow_parameters[patient,0]==1:
            HR=77
        elif inflow_parameters[patient,0]==2:
            HR=77+9.1
            
        if inflow_parameters[patient,1]==0:
            SV=58.7-11.1
        elif inflow_parameters[patient,1]==1:
            SV=58.7
        elif inflow_parameters[patient,1]==2:
            SV=58.7+11.1

        if inflow_parameters[patient,2]==0:
            LVET=282-23
        elif inflow_parameters[patient,2]==1:
            LVET=282
        elif inflow_parameters[patient,2]==2:
            LVET=282+23                

        if inflow_parameters[patient,3]==0:
            PFT=80-0
        elif inflow_parameters[patient,3]==1:
            PFT=80
        elif inflow_parameters[patient,3]==2:
            PFT=80+0                

        if inflow_parameters[patient,4]==0:
            RFV=0.7-0
        elif inflow_parameters[patient,4]==1:
            RFV=0.7
        elif inflow_parameters[patient,4]==2:
            RFV=0.7+0  

        if inflow_parameters[patient,5]==0:
            MAV=570-128
        elif inflow_parameters[patient,5]==1:
            MAV=570
        elif inflow_parameters[patient,5]==2:
            MAV=570+128  
            
            
    elif Agecase==65:
        if inflow_parameters[patient,0]==0:
            HR=76.3-9
        elif inflow_parameters[patient,0]==1:
            HR=76.3
        elif inflow_parameters[patient,0]==2:
            HR=76.3+9
            
        if inflow_parameters[patient,1]==0:
            SV=55.8-10.4
        elif inflow_parameters[patient,1]==1:
            SV=55.8
        elif inflow_parameters[patient,1]==2:
            SV=55.8+10.4

        if inflow_parameters[patient,2]==0:
            LVET=282-23
        elif inflow_parameters[patient,2]==1:
            LVET=282
        elif inflow_parameters[patient,2]==2:
            LVET=282+23                

        if inflow_parameters[patient,3]==0:
            PFT=80-0.1
        elif inflow_parameters[patient,3]==1:
            PFT=80
        elif inflow_parameters[patient,3]==2:
            PFT=80+0.1              

        if inflow_parameters[patient,4]==0:
            RFV=0.8-0.1
        elif inflow_parameters[patient,4]==1:
            RFV=0.8
        elif inflow_parameters[patient,4]==2:
            RFV=0.8+0.1  

        if inflow_parameters[patient,5]==0:
            MAV=568-119
        elif inflow_parameters[patient,5]==1:
            MAV=568
        elif inflow_parameters[patient,5]==2:
            MAV=568+119              

    elif Agecase==75:
        if inflow_parameters[patient,0]==0:
            HR=74.4-9
        elif inflow_parameters[patient,0]==1:
            HR=74.4
        elif inflow_parameters[patient,0]==2:
            HR=74.4+9
            
        if inflow_parameters[patient,1]==0:
            SV=53.6-9.8
        elif inflow_parameters[patient,1]==1:
            SV=53.6
        elif inflow_parameters[patient,1]==2:
            SV=53.6+9.8

        if inflow_parameters[patient,2]==0:
            LVET=282-23
        elif inflow_parameters[patient,2]==1:
            LVET=282
        elif inflow_parameters[patient,2]==2:
            LVET=282+23                

        if inflow_parameters[patient,3]==0:
            PFT=80-0.2
        elif inflow_parameters[patient,3]==1:
            PFT=80
        elif inflow_parameters[patient,3]==2:
            PFT=80+0.2                

        if inflow_parameters[patient,4]==0:
            RFV=0.8-0.1
        elif inflow_parameters[patient,4]==1:
            RFV=0.8
        elif inflow_parameters[patient,4]==2:
            RFV=0.8+0.1

        if inflow_parameters[patient,5]==0:
            MAV=568-122
        elif inflow_parameters[patient,5]==1:
            MAV=568
        elif inflow_parameters[patient,5]==2:
            MAV=568+122  
    
    output_param=np.array([Agecase,HR/60,SV*1e-6,LVET*1e-3,PFT*1e-3,RFV*1e-6,MAV*133.322])
    return output_param



for patient in range(0,1):
    
    
    number=(patient+1)*6
    i=patient*6
    
    
    T=1 #s
    dT=5e-6 #s 
    
    lowertime=40 #min
    uppertime=lowertime+T
    maxtime=uppertime+(T/2)
    Timenumber=maxtime/dT
    PI=3.142
    
    Age=np.linspace(25, 75, num=(number-i))
    
    Psys=17300 #Pa
    Pdia=10100 #Pa
    
    L=np.array([40,20,34,39,208,177,34,156,34,177,177,177,177,148,422,422,
                148,29,
                
                5,15,10,5,20,5,27,22,30,15,25,55,63,
                
                5,7,10,5,20,5,27,22,30,15,25,55,63,
                
                5,5,12,12,3,
                
                119,34,34,19,66,61,19,33,68,50,50,50,50,33,43,65,90,90,
                
                119,34,34,19,66,61,19,33,68,50,50,50,50,33,43,65,90,90,
                
                43,43,80,50,20,20,112,102,5,5,5,5,5,5,84,84,70,56,41,41,21,7,
                
                43,43,80,50,20,20,112,102,5,5,5,5,5,5,84,84,70,56,41,41,21,7,7
                
                ])/1000
    
    E=np.array([0.4,0.8,1.6])*1e6
    
    
    Ao=np.zeros(len(L))
    
    
    while i < number:
        
        
        Agecase=Age
        Lcase=L
        
        
        Ao[0]=ata.ascending(Agecase)
        Ao[1]=ata.arch(Agecase)
        Ao[2]=ata.BCT(Agecase)
        Ao[3]=ata.arch(Agecase)
        Ao[4]=ata.CCA(Agecase)
        Ao[5]=ata.CCA(Agecase)
        Ao[6]=ata.LS(Agecase)
        Ao[7]=ata.descending(Agecase)
        Ao[8]=ata.LS(Agecase)
        Ao[9]=ata.ICAL(Agecase)
        Ao[10]=ata.ICAL(Agecase)
        Ao[11]=ata.ICAR(Agecase)
        Ao[12]=ata.ICAR(Agecase)
        Ao[13]=ata.VAR(Agecase)
        Ao[14]=PI*(0.403)*(0.403)/(100*100)
        Ao[15]=PI*(0.403)*(0.403)/(100*100)
        Ao[16]=ata.VAL(Agecase)
        Ao[17]=ata.BA(Agecase)
        
        Ao[18]=ata.PCA(Agecase)
        Ao[19]=ata.PComA(Agecase)
        Ao[20]=ata.PCA(Agecase)
        Ao[21]=(Ao[20]*1.7)/2.1
        Ao[22]=(Ao[20]*1)/2.1
        Ao[23]=(Ao[20]*1.7)/2.1
        Ao[24]=(Ao[20]*1.3)/2.1
        Ao[25]=(Ao[20]*1.5)/2.1
        Ao[26]=(Ao[20]*1.6)/2.1
        Ao[27]=(Ao[20]*1.4)/2.1
        Ao[28]=(Ao[20]*1.2)/2.1
        Ao[29]=(Ao[20]*1.5)/2.1
        Ao[30]=(Ao[20]*0.8)/2.1
        
        Ao[31]=ata.PCA(Agecase)
        Ao[32]=ata.PComA(Agecase)
        Ao[33]=ata.PCA(Agecase)
        Ao[34]=(Ao[20]*1.7)/2.1
        Ao[35]=(Ao[20]*1)/2.1
        Ao[36]=(Ao[20]*1.7)/2.1
        Ao[37]=(Ao[20]*1.3)/2.1
        Ao[38]=(Ao[20]*1.5)/2.1
        Ao[39]=(Ao[20]*1.6)/2.1
        Ao[40]=(Ao[20]*1.4)/2.1
        Ao[41]=(Ao[20]*1.2)/2.1
        Ao[42]=(Ao[20]*1.5)/2.1
        Ao[43]=(Ao[20]*0.8)/2.1
        
        Ao[44]=ata.ICAL(Agecase)
        Ao[45]=ata.ICAR(Agecase)
        Ao[46]=ata.ACA(Agecase)
        Ao[47]=ata.ACA(Agecase)
        Ao[48]=ata.AComA(Agecase)
        
        Ao[49]=ata.MCA(Agecase)
        Ao[50]=(Ao[49]*1.7)/2
        Ao[51]=(Ao[49]*1.7)/2
        Ao[52]=(Ao[49]*1.2)/2
        Ao[53]=(Ao[49]*1.4)/2
        Ao[54]=(Ao[49]*1.1)/2
        Ao[55]=(Ao[49]*1.2)/2
        Ao[56]=(Ao[49]*1.1)/2
        Ao[57]=(Ao[49]*1.7)/2
        Ao[58]=(Ao[49]*1.5)/2
        Ao[59]=(Ao[49]*1.5)/2
        Ao[60]=(Ao[49]*1.4)/2
        Ao[61]=(Ao[49]*1.4)/2
        Ao[62]=(Ao[49]*1.4)/2
        Ao[63]=(Ao[49]*1.5)/2
        Ao[64]=(Ao[49]*1.4)/2
        Ao[65]=(Ao[49]*1.5)/2
        Ao[66]=(Ao[49]*1.5)/2
        
        Ao[67]=ata.MCA(Agecase)
        Ao[68]=(Ao[49]*1.7)/2
        Ao[69]=(Ao[49]*1.7)/2
        Ao[70]=(Ao[49]*1.2)/2
        Ao[71]=(Ao[49]*1.4)/2
        Ao[72]=(Ao[49]*1.1)/2
        Ao[73]=(Ao[49]*1.2)/2
        Ao[74]=(Ao[49]*1.1)/2
        Ao[75]=(Ao[49]*1.7)/2
        Ao[76]=(Ao[49]*1.5)/2
        Ao[77]=(Ao[49]*1.5)/2
        Ao[78]=(Ao[49]*1.4)/2
        Ao[79]=(Ao[49]*1.4)/2
        Ao[80]=(Ao[49]*1.4)/2
        Ao[81]=(Ao[49]*1.5)/2
        Ao[82]=(Ao[49]*1.4)/2
        Ao[83]=(Ao[49]*1.5)/2
        Ao[84]=(Ao[49]*1.5)/2
        
        Ao[85]=Ao[46]
        Ao[86]=Ao[46]
        Ao[87]=(Ao[85]*1.7)/2
        Ao[88]=(Ao[85]*1.5)/2
        Ao[89]=(Ao[85]*1.3)/2
        Ao[90]=(Ao[85]*1.3)/2
        Ao[91]=(Ao[85]*1.2)/2
        Ao[92]=(Ao[85]*1.3)/2
        Ao[93]=(Ao[85]*1.9)/2
        Ao[94]=(Ao[85]*1.9)/2
        Ao[95]=(Ao[85]*1.9)/2
        Ao[96]=(Ao[85]*1.9)/2
        Ao[97]=(Ao[85]*1.9)/2
        Ao[98]=(Ao[85]*1.9)/2
        Ao[99]=(Ao[85]*1.4)/2
        Ao[100]=(Ao[85]*1.4)/2
        Ao[101]=(Ao[85]*1.3)/2
        Ao[102]=(Ao[85]*1.2)/2
        Ao[103]=(Ao[85]*1.3)/2
        Ao[104]=(Ao[85]*1.7)/2
        Ao[105]=(Ao[85]*1.4)/2
        Ao[106]=(Ao[85]*1.1)/2
        
        Ao[107]=Ao[46]
        Ao[108]=Ao[46]
        Ao[109]=(Ao[85]*1.7)/2
        Ao[110]=(Ao[85]*1.5)/2
        Ao[111]=(Ao[85]*1.3)/2
        Ao[112]=(Ao[85]*1.3)/2
        Ao[113]=(Ao[85]*1.2)/2
        Ao[114]=(Ao[85]*1.3)/2
        Ao[115]=(Ao[85]*1.9)/2
        Ao[116]=(Ao[85]*1.9)/2
        Ao[117]=(Ao[85]*1.9)/2
        Ao[118]=(Ao[85]*1.9)/2
        Ao[119]=(Ao[85]*1.9)/2
        Ao[120]=(Ao[85]*1.9)/2
        Ao[121]=(Ao[85]*1.4)/2
        Ao[122]=(Ao[85]*1.4)/2
        Ao[123]=(Ao[85]*1.3)/2
        Ao[124]=(Ao[85]*1.2)/2
        Ao[125]=(Ao[85]*1.3)/2
        Ao[126]=(Ao[85]*1.7)/2
        Ao[127]=(Ao[85]*1.4)/2
        Ao[128]=(Ao[85]*1.1)/2
        
        Ao[129]=ata.PComA(Agecase)
        
        
        Aocase=Ao
         
        a = 0.2802 #
        b = -0.5053*1000 #m-1
        c = 0.1324 #
        d = -0.01114*1000 #m-1
        r0=np.zeros(len(Lcase))
        h=np.zeros(len(Lcase))
        for j in range(0,len(Lcase)):
            r0[j]=np.sqrt(Aocase[j]/PI)
            h[j]=r0[j]*[a*np.exp(b*r0[j])+c*np.exp(d*r0[j])][0]
            
            
        param=parameters(patient,Agecase)   
        name='param'+str(patient)+'_PCAR2'
        np.save(name,param)        
        
        Rtot=((1/3)*Psys+(2/3)*Pdia)/(param[2]*param[1])
        Ctot = 1.34/Rtot
    
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
        

    
        for j in [0]:        
            
       
                
            name='CoW_'+str(i)+'_PCAR2'+str(j)+'.in'
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
    q = 2
q  0
    q = 2
{1}  {2}  {3}
{1}  {2}  {3}\n'''.format(k+1,outlet[k,1],int(outlet[k,2]),int(outlet[k,3])))
                elif outlet[k,1]=='W':
                    f.write('''{1}  {2}  {3} 	 # Domain {0}
{1}  {2}  {3}\n'''.format(k+1,inlet[k,1],int(inlet[k,2]),int(inlet[k,3])))
                    f.write('''W  {:.2e}
W  {:.2e}\n'''.format(C[k],Rt[k]))
                elif outlet[k,1]=='T':
                    f.write('''{1}  {2}  {3} 	 # Domain {0}
{1}  {2}  {3}\n'''.format(k+1,inlet[k,1],int(inlet[k,2]),int(inlet[k,3])))
                    f.write('''T  1
T 1\n''')
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
