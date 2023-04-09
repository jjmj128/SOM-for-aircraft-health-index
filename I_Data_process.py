# -*- coding: utf-8 -*-
"""
"""

""" Modules """
import sys
import time
import csv
import numpy as np
import random as rd


"""
Functions
"""
def pathGeneration(input_dir,input_file,extention='.csv',debug=False):
     """
     """
     if(input_dir == ''):
          input_dir = input("Directory path: ")
     if(input_file == ''):
          input_file = input("Filename in {} directory: ".format(input_dir))

     if(extention not in input_file):
          input_file = input_file + extention
     input_path =  input_dir + '/' + input_file  

     if(debug):
          print("Path of the file selected: {}".format(input_path))

     return input_path



def txtToCsv(input_path,output_path,csv_delimiter=';',debug=False):
     """
     """
     print("txtToCsv generating ({})...".format(input_path))

     input_file, txt_file = openFile(input_path)
     output_file, out_csv = writeFile(output_path, csv_delimiter)

     for row in txt_file:
          row_tmp = row[0].split(' ')
          while(row_tmp[-1] == ''):
               row_tmp = row_tmp[:-1] #remove empty value
          out_csv.writerow(row_tmp)

     output_file.close()
     input_file.close()

     print("Csv file generated ({})!".format(output_path))



def columnExtraction(input_path,output_path,selected_column,csv_delimiter=';',debug=False):
     """
     """
     print("Column extracting ({})...".format(input_path))

     input_file, txt_file = openFile(input_path, csv_delimiter)
     output_file, out_csv = writeFile(output_path, csv_delimiter)

     for row in txt_file:
          row_tmp = []
          for item in selected_column:
               row_tmp.append(row[item])
          out_csv.writerow(row_tmp)

     output_file.close()
     input_file.close()

     print("Column extracted and file generated ({})!".format(output_path))



def rowExtraction(input_path,output_path,selected_row,selected_col,csv_delimiter=';',debug=False):
     """
     """
     print("Row extracting ({})...".format(input_path))

     input_file, txt_file = openFile(input_path,csv_delimiter)
     output_file, out_csv = writeFile(output_path,csv_delimiter)

     for row in txt_file:
          if(float(row[selected_col]) == float(selected_row)):
               out_csv.writerow(row)

     output_file.close()
     input_file.close()

     print("Row extracted and file generated ({})!".format(output_path))



def openFile(file_path,file_delemiter=';',debug=False):
     """
     """
     print("Opening file ({})...".format(file_path))
     try:
          file_instance = open(file_path)
          file_content = csv.reader(file_instance, delimiter=file_delemiter)
          print("File opened!")
          return file_instance, file_content
     except IOError:
          print("Input file {} not founded...".format(file_path))
          sys.exit()
     except:
          print("Unexpected error: ", sys.exc_info()[0])
          raise



def writeFile(file_path,file_delemiter=';',chmod='w',debug=False):
     """
     """
     try:
          file_instance = open(file_path, chmod, newline='')
          file_content = csv.writer(file_instance,delimiter=file_delemiter)
          return file_instance, file_content
     except IOError as e:
          if(e.errno == 13):
               print("The file {} is already open !".format(file_path))
          else:
               print(e.strerror)
               print("Input file {} not founded...".format(file_path))
          sys.exit()
     except:
          print("Unexpected error: ", sys.exc_info()[0])
          raise



def getFile(file_path,file_delemiter=';',selected_column=None,debug=False):
     """
     """
     print("Reading file ({})...".format(file_path))
     try:
          out = np.genfromtxt(file_path, delimiter=file_delemiter, usecols=selected_column)
          print("File loaded!")
          return out
     except IOError:
          print("Input file {} not founded...".format(file_path))
          sys.exit()
     except:
          print("Unexpected error: ", sys.exc_info()[0])
          raise



def saveFile(file_path,data,file_delemiter=';',chmod='w',debug=False):
     """
     """
     print("File save ({})...".format(file_path))
     try:
          file_instance, file_content = writeFile(file_path,file_delemiter)
          for row in data:
               file_content.writerow(row)
          file_instance.close()
          print("File saved!")
          return 0
     except IOError:
          print("Input file {} not founded...".format(file_path))
          sys.exit()
     except:
          print("Unexpected error: ", sys.exc_info()[0])
          raise



def dataMinMax(data,debug=False):
     """
     Args: data (n x m)
     return: data_min (list), data_max (list)
     """
     if(type(data) == list):
          data_min = [min(data[0])]
          data_max = [max(data[0])]
     else:
          _, nbCol = data.shape
          data_min = []
          data_max = []
          for i in range(nbCol):
               data_min.append(data[:,i].min())
               data_max.append(data[:,i].max())
     return data_min, data_max



def dataNorm(norm_type, data, data_norm_info=[], num_col_start=0, num_col_end=-1, debug=False):
     """ Normalize the data """
     if(debug):
          print("Data normalization ({})...".format(norm_type))
     if(num_col_end == -1):
          num_col_end = data.shape[1] - num_col_start - 1
     norm_info = []
     if(norm_type == 'rescaling'):
          if(len(data_norm_info)>0):
               if(debug): print("Use existing norm info")
               data_min, data_max = data_norm_info
          else:          
               data_min, data_max = dataMinMax(data)
          data_bound = np.array(list((data_min,data_max))) # matrix [min;max]
          norm_info.append(list((data_min,data_max)))
          norm_info = norm_info[0]

          data_min = data_bound[0,:]
          data_max = data_bound[1,:]
          
          for i in range(num_col_start,num_col_end+1):
               delta = (data_max[i] - data_min[i])
               if(delta == 0):
                    delta = 1
               data[:,i] = (data[:,i] - data_min[i]) / delta
     elif(norm_type == 'stand'):
          for i in range(num_col_start,num_col_end+1):
               feature = data[:,i]
               if(len(data_norm_info)>0):
                    mu, sgm = data_norm_info[i-num_col_start]
               else:
                    mu = np.mean(feature)
                    sgm = np.std(feature)
                    if(sgm == 0):
                         sgm = 1
               data[:,i] = (feature - mu) / sgm
               norm_info.append([mu,sgm])
     else:
          print("unkown parameter norm_type: {}".format(norm_type))
          sys.exit()
     if(debug):
          print("Data normalized!")
     return data, norm_info



def dataNormMode(norm_type, data, mode_norm_info=[], num_col_start=0, num_col_end=-1, mode_col=-1, debug=False):
     if(debug):
          print("Data by mode normalization ({})...".format(norm_type))
     if(num_col_end == -1):
          num_col_end = data.shape[1] - num_col_start
     if(num_col_end == -1 and mode_col == -1):
          num_col_end -= 1
          print("TBC dataNormMode!!!")
     data_norm_info = []
     nbMode = int(max(data[:,mode_col])) #-1 is the last column
     for i in range(nbMode):
          mode_i = i + 1
          tmp = data[data[:,mode_col]==mode_i,:]
          if(len(mode_norm_info) > 0):
               tmp_mode_norm_info = mode_norm_info[i]
          else:
               tmp_mode_norm_info = []
          tmp, tmp_norm_info = dataNorm(norm_type, tmp, tmp_mode_norm_info, num_col_start, num_col_end, debug)
          data[data[:,mode_col]==mode_i,:] = tmp
          if(len(mode_norm_info) == 0):
               data_norm_info.append(tmp_norm_info)
     if(debug):
          print("Data by mode normalized!")     
     return data, data_norm_info



def timeFormat(start_time, end_time):
     delta_time = end_time - start_time
     h_time = int(delta_time/3600)
     delta_time = delta_time - h_time
     min_time = int(delta_time/60)
     delta_time = delta_time - min_time
     s_time = round(delta_time)
     return h_time, min_time, s_time


def dataFusion(csv_delimiter=';',debug=False):
     """
     """

     train_engine_pers = 80
     train_output_path = pathGeneration(output_dir, 'train_FD00XCsv')
     train_output_file, train_out_csv = writeFile(train_output_path,csv_delimiter)
     test_output_path = pathGeneration(output_dir, 'test_FD00XCsv')
     test_output_file, test_out_csv = writeFile(test_output_path,csv_delimiter)     
     file_liste = ['train_FD001Csv','train_FD002Csv','train_FD003Csv','train_FD004Csv']
     file_engineNb = [100, 260, 100, 249]
     engine = 0
     unit_max = sum(file_engineNb)
     
     train_engine_nb = int(train_engine_pers/100.*unit_max) #Number of engine for WB model
     #test_engine_nb = unit_max - train_engine_nb #Number of engine for BB model
     train_test_engine_rd = rd.sample(range(1,unit_max+1), unit_max) #Generate vector of random engine
     train_engine_rd = train_test_engine_rd[:train_engine_nb] #Get engine label
     test_engine_rd = train_test_engine_rd[train_engine_nb:] #Get engine label
     
     print("test engine: ", test_engine_rd)
     
     for file in file_liste:
          input_path = pathGeneration(input_dir, file)
          input_file, txt_file = openFile(input_path, csv_delimiter)

          engine = engine + 1
          engine_i_tmp = 1
          for row in txt_file:
               row_tmp = []
               engine_i = int(row[0])
               if(engine_i != engine_i_tmp):
                    engine_i_tmp = engine_i
                    engine = engine + 1
               row_tmp.append(str(engine))
               for item in row[1:]:
                    row_tmp.append(item)
               if(engine in train_engine_rd):
                    train_out_csv.writerow(row_tmp)
               else:
                    test_out_csv.writerow(row_tmp)
     
          input_file.close()
     train_output_file.close()
     test_output_file.close()
     print("File generated ({})!".format(train_output_path))
     print("File generated ({})!".format(test_output_path))


#dataFusion()


""" Settings """
#User settings
debug = True
norm_type = 'rescaling' # rescaling: rescaling between [0,1] | stand: Standardization (x-mean)/standard deviation
#input_file = 'train'
#input_dir = 'input'
input_file = 'traincruise' #train_FD002  test_FD002
input_dir = 'train'

#input_dir = 'ksenia'
#input_file = 'gpu2' #'cars'

normalizeOnExisting = False #Take existing normalize info to normalize data


output_file = input_file

input_file_featureExtraction = output_file
output_file_featureExtraction = input_file_featureExtraction+'Feature'

input_file_dataNormalized = output_file_featureExtraction
output_file_dataNormalized = input_file_dataNormalized+'Norm'
output_file_dataNormalizedInfo = output_file_dataNormalized+'Info'

input_file_dataNormalizedInv = input_file_dataNormalized
output_file_dataNormalizedInv = input_file_dataNormalizedInv+'Norm'

input_file_flightExtraction = output_file_dataNormalized
output_file_flightExtraction = input_file_flightExtraction+'Flight'

selected_flight = [i+1 for i in range(10)]

selected_column = (0,1,8,12,15) # 

#Defaul settings
output_dir = input_dir
input_dir_featureExtraction = input_dir
output_dir_featureExtraction = input_dir
input_dir_flightExtraction = input_dir
output_dir_flightExtraction = input_dir
input_dir_dataNormalized = input_dir
output_dir_dataNormalized = input_dir

csv_delimiter = ';'


"""
Main program
"""
if __name__ == "__main__":
     """ Init """
     print("### Program starting... ###\n")
     start_time=time.time()
     
     """Generate path"""
     input_path = pathGeneration(input_dir, input_file, '.csv')
     output_path = pathGeneration(output_dir, output_file)
     
     """Convert txt to csv file"""
     #txtToCsv(input_path, output_path)
     
     
     
     """Generate  path of feature"""
     input_path_featureExtraction = pathGeneration(input_dir_featureExtraction, input_file_featureExtraction)
     output_path_featureExtraction = pathGeneration(output_dir_featureExtraction, output_file_featureExtraction)
     """Extract feature from column selection"""
     columnExtraction(input_path_featureExtraction, output_path_featureExtraction, selected_column)
     
     
     
     """ Open file to be dataNormalized """
     input_path_dataNormalized = pathGeneration(input_dir_dataNormalized, input_file_dataNormalized)
     data = getFile(input_path_dataNormalized)
     
     """ Get boundaries """
     data_min, data_max = dataMinMax(data)
     data_bound = np.array(list((data_min,data_max))) # matrix [min;max]
     
     """ Previously normalized data """
     if(normalizeOnExisting):
          output_path_dataNormalizedInfo = pathGeneration(output_dir_dataNormalized, output_file_dataNormalizedInfo.replace('test','train'))
          _, dataNormInfoTmp = openFile(output_path_dataNormalizedInfo)
          norm_type = ''
          dataNormInfo = []
          for item in dataNormInfoTmp:
               if(norm_type !=''):
                    dataNormInfo.append(item)
               if(item[0]=='s'): #stand norm
                    norm_type='stand'
                    dataNormInfo = []
               if(item[0]=='r'):
                    norm_type='rescaling'
                    dataNormInfo = []                
          dataNormInfo = np.array(dataNormInfo).astype(float)
     else:
          dataNormInfo = []
     """ Normalize data """
     num_col_start = 2
     data, norm_info = dataNorm(norm_type, data, dataNormInfo, num_col_start, len(selected_column)-1)
     """ Save data normalized """
     output_path_dataNormalized = pathGeneration(output_dir_dataNormalized, output_file_dataNormalized)
     saveFile(output_path_dataNormalized, data)
     """ Save normalized data info """
     output_path_dataNormalizedInfo = pathGeneration(output_dir_dataNormalized, output_file_dataNormalizedInfo)
     saveFile(output_path_dataNormalizedInfo, [i for j in [[norm_type], norm_info] for i in j])     
     
     
     
     """Flight extraction"""
     #input_path_flightExtraction = pathGeneration(input_dir_flightExtraction, input_file_flightExtraction)
     #for i in selected_flight:
          #output_file_flightExtraction_tmp = output_file_flightExtraction+str(i)
          ##Generate  path of feature
          #output_path_flightExtraction = pathGeneration(output_dir_flightExtraction, output_file_flightExtraction_tmp)
          ##Flight Extraction
          #rowExtraction(input_path_flightExtraction, output_path_flightExtraction, selected_row=i, selected_col=0)
     
     
     end_time = time.time()
     h_time, min_time, s_time = timeFormat(start_time, end_time)
     print("\n### End of execution in {} h {} min {} s ###".format(h_time,min_time,s_time))
     print("\n")