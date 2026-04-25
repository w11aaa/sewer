import hashlib as Hashlib
import imageio as ImageIO
import shutil as Shutil
import tempfile as Temp
import streamlit as App
import cv2 as CV2
import os as OS
import numpy as NP
import plotly.express as Chart
import matplotlib.pyplot as Plt
import torch as Torch
import zipfile as Zip
from PIL import Image
from ultralytics import YOLO
from io import BytesIO as BIO
from streamlit_pdf_viewer import pdf_viewer as PDF
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.drawing.image import Image as XLImage
from pandas import Series, DataFrame, cut
from openpyxl.worksheet.table import Table
from openpyxl.worksheet.table import TableStyleInfo
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table as PDFTable, TableStyle, Image as PDFImage, Paragraph, Spacer
from reportlab.lib.units import inch