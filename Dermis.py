from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Interface():
    
    def __init__(self):
        
        try:
            Interface.CarregarModelo(self)
        except:
            pass
        self.root = Tk()

        self.root.title("D E R M I S")
        
        Button(self.root, text='Selecione a imagem', command = self.CapturaImagem).grid(row=0,column=0, pady=5)
        
        Button(self.root, text='Treinar rede', command = self.TreinaModelo, width=10, height=2).grid(row=0,column=1)
        
        Button(self.root, text='Classificar', command = self.ClassificarImagens, width = 10, height = 2).grid(row=1, column = 1)
        
        Button(self.root, text='Salvar modelo', command= self.SalvarModelo, width=10, height=2).grid(row=2, column=1)

        self.root.mainloop()
        
    def CapturaImagem(self):
        self.filename = askopenfilename()
        
        self.image = Image.open(self.filename)
        self.image = self.image.resize((400,400), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)
        
        label = Label(self.root, image = self.photo).grid(row=1,column = 0, padx = 15, pady = 5, rowspan= 3)
    
    def TreinaModelo(self):
        self.rede = Sequential()
        
        self.rede.add(Conv2D(64, (3,3), input_shape = (64,64,3), activation='relu'))
        self.rede.add(BatchNormalization())
        self.rede.add(MaxPooling2D(pool_size = (2,2)))
        
        self.rede.add(Flatten())


        
        self.rede.add(Dense(units=100, activation='relu'))
        self.rede.add(Dense(units=100, activation='relu'))
        self.rede.add(Dense( units=5, activation='softmax'))
        
        self.rede.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        gerador_treinamento = ImageDataGenerator(rescale=1. / 255, rotation_range = 7, horizontal_flip = True, shear_range = 0.2, height_shift_range = 0.07, zoom_range = 0.2)
        
        gerador_teste = ImageDataGenerator(rescale= 1. / 255, )
        
        base_treinamento = gerador_treinamento.flow_from_directory('Dataset/Treinamento', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
        base_teste = gerador_teste.flow_from_directory('Dataset/Teste', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
        
        self.rede.fit_generator(base_treinamento, steps_per_epoch=20, epochs=3, validation_data = base_teste, validation_steps = 20)
        
        
        
    def ClassificarImagens(self):
       
        
        imagem_teste = image.load_img(self.filename, target_size = (64,64))
        
        imagem_teste = image.img_to_array(imagem_teste)
        
        imagem_teste /= 255
        
        imagem_teste = np.expand_dims(imagem_teste, axis = 0)
        
        previsao = self.rede.predict(imagem_teste)
        
        print(previsao)
        resultado = np.argmax(previsao,axis=1)
        
        Classes = ['Cicatriz','Sifilis','Melanoma','Pele normal','Verruga']

        
        Variavel = str(Classes[resultado[0]])
        
        Label(self.root, text=Variavel).grid(row=1,column=0)
        
            
    def SalvarModelo(self):
        model = self.rede
        model_json = model.to_json()
        with open('modelTwo.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights("modelTwo.h5")
        print("Modelo salvo no disco")
            
    def CarregarModelo(self):
        json_file = open('modelTwo.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        loaded_model.load_weights("modelTwo.h5")
        print("Modelo carregado com sucesso")
        
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.rede = loaded_model
        
    

        
Interface()