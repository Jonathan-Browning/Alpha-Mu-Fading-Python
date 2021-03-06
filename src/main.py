# -*- coding: utf-8 -*-
"""
@author: Jonathan Browning
"""
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import time
from Class.alphamu import AlphaMu
from tkinter import messagebox
import os

def draw_envelope(data):
    plotname = "envelope_plot.png"
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(1)
    plt.xlabel(r"$r$", fontsize=18)
    plt.ylabel(r"$f_{R}(r)$", fontsize=18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18) 
    plt.xlim((0, 6))
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tick_params(direction='in')
    plt.plot(data.r, data.envelopeProbability, "k", label='Theoretical')  
    plt.plot(data.xdataEnv[1:len(data.xdataEnv):2], data.ydataEnv[1:len(data.ydataEnv):2], "k.", label='Simulation') 
    leg = plt.legend(fontsize=15)
    leg.get_frame().set_edgecolor('k')
    plt.savefig(plotname)
    plt.close(1)
    return plotname


def main():   
        
    # Setting up window layout
    layout = [[sg.Text(r'Please enter \alpha, \mu and \hat{r}', font='Helvetica 18')],      
          [sg.Text(r"\alpha:", size=(8, 1), font='Helvetica 18'), sg.Input(key='-alpha', size=(5, 1), font='Helvetica 18')],      
          [sg.Text("\mu:", size=(8, 1), font='Helvetica 18'), sg.Input(key='-mu', size=(5, 1), font='Helvetica 18')], 
          [sg.Text(r"\hat{r}:", size=(8, 1), font='Helvetica 18'), sg.Input(key=r'-\hat{r}', size=(5, 1), font='Helvetica 18')],      
          [sg.Button('Calculate', font='Helvetica 18'), sg.Exit(font='Helvetica 18')],
          [sg.Text("Time (s):", size=(8, 1), font='Helvetica 18'), sg.Txt('', size=(8,1), key='output')],
          [sg.Image(key='-Image1')]]
        
    window = sg.Window(r"The \alpha-\mu fading model", layout, finalize=True, font='Helvetica 18')

    # The Event Loop                 
    while True:
        # Reading user inputs
        event, values = window.read() 
        
        # Close if the exist button is pressed or the X
        if event in (sg.WIN_CLOSED, 'Exit'):
            break      
        
        # Rice class instance which calculates everything and time the exeuction
        start = time.time()
        try:
            s = AlphaMu(values['-alpha'], values['-mu'], values['-\hat{r}'])
        except Exception as e:  # displays the error message and will force the program to close
            messagebox.showerror("Error", e)
            continue
        end = time.time()
        
        # update the execution time
        exeTime = round(end - start, 4) # Roudn to 4 decimal places
        window['output'].update(exeTime)  # Display the execution time 
        
        # drawing the figures
        imageFileName1 = draw_envelope(s)
        window.Element('-Image1').Update(imageFileName1)
        os.remove(imageFileName1) # need to remove the image file again
        
    window.close()
    
if __name__ == "__main__":
    main()