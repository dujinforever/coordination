from Code.Step_code1 import *
class Coordination(object):
    
    def __init__(self,filename,b_name,b_clusters,b_Marker,save_path='   '):
        self.filename = filename
        self.b_name = b_name
        self.b_clusters = b_clusters
        self.b_Marker = b_Marker
        self.constName_joint = ['腰椎屈曲,deg', 'Lumbar Lateral - RT,deg', 'Lumbar Axial - RT,deg',
       '胸椎屈曲,deg', 'Thoracic Lateral - RT,deg', 'Thoracic Axial - RT,deg','肘关节屈曲 右,deg', 
       '肩屈曲 右,deg', '肩外展 右,deg',
       'Shoulder Rotation - out 右,deg', '伸腕 右,deg',
       '手腕径向 右,deg','Wrist Supination 右,deg',
        '髋关节屈曲 右,deg', '髋关节外展 右,deg',
        'Hip Rotation - out 右,deg', 
       '膝关节屈曲 右,deg', '踝关节背屈 右,deg', 
       '踝关节反转 右,deg', '踝关节外展 右,deg']
        self.cal_joint_table()
        self.knee_ankle = Compare_Mul_Locomotion(self.joint_table,'膝关节屈曲 右,deg','踝关节背屈 右,deg')
        self.hip_knee = Compare_Mul_Locomotion(self.joint_table,'髋关节屈曲 右,deg','膝关节屈曲 右,deg')
        self.savepath = save_path	

    def cal_joint_table(self):
        datalist = main.loaddata(self.filename,type = 'pd')
        array = [0,1,2,3,4,5,6,7,8,9]
        Propertytable = []
        for i in array: 
            MulCycTable = datalist[i][1]
            Curve = MulCycTable[self.b_name[i]]
            MulCycTable = MulCycTable[self.constName_joint]
            MulCycTable = pd.DataFrame(main.Buttterworth(MulCycTable,100,6,'low'))
            MulCycTable.columns = self.constName_joint
            Index = CycleIndex(Curve, n_Marker = self.b_Marker[i], n_clusters = self.b_clusters[i])
            CycleTable,PropertyTable,NonCycleDataFrame = Cycle_n(MulCycTable,Index)
            PropertyTabletemp = {}
            for i in self.constName_joint:
                PropertyTabletemp[i] = PropertyTable[i]
            Propertytable.append(PropertyTabletemp)
            self.joint_table = Propertytable
        return self.joint_table

    def cal_CRP_table(self,name1,name2):
        CRP = []
        for i in range(10):
            table1 = self.joint_table[i][name1]
            table2 = self.joint_table[i][name2]
            table1_PhaseAngle = cal_PhaseAngle(table1)
            table2_PhaseAngle = cal_PhaseAngle(table2)
            CRP.append(pd.DataFrame(table1_PhaseAngle[0] - table2_PhaseAngle[0]))
        self.CRP_table =  CRP
        return CRP

    def cal_result(self,name1,name2):
        result = Compare_Mul_Locomotion(self.joint_table,name1,name2)
        return result

    def plot_Phase_fig(self):
        for key in self.constName_joint:#dict类型，索引为关节角度
            f,axes = plt.subplots(5,2)
            for i in range(len(self.joint_table)):
                row = math.floor(i/2)
                col = int(math.fmod(i,2))
                value = cal_PhaseAngle(self.joint_table[i][key])
                axes[row,col].plot(value[0])
                #[row,col].set_title(key)
            f.savefig('.//'+self.savepath+'//PhaseAngle//'+key+'.png',dpi = 600)
            plt.pause(1)
            plt.close()
    def plot_CRP_angle(self,result_value,save_fig = False):
        f,axes = plt.subplots(5,2)
        for i in  range(len(result_value[0])):
            row = math.floor(i/2)
            col = int(math.fmod(i,2))
            axes[row,col].plot(result_value[0][i],'r')
            ax2 = axes[row,col].twinx()#绘制双坐标
            ax2.plot(np.array(self.joint_table[i]['膝关节屈曲 右,deg'])[:,1],'g--')
            ax2.plot(np.array(self.joint_table[i]['髋关节屈曲 右,deg'])[:,1],'g')
            if save_fig == True:
                f.savefig('.//'+self.savepath+'//result_fig//CRP'+result_str[j]+'.png',dpi = 600)
        #for i in  range(len(result_value[1])):
          #  row = math.floor(i/2)
           # col = int(math.fmod(i,2))
           # axes[row,col].plot(result_value[1][i])
              #  f.savefig('.//Result-data-zhouqingyang//result_fig//CV'+result_str[j]+'.png',dpi = 600)