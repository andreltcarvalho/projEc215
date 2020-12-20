import cv2, time
from datetime import datetime

fundo_estatico = None

time = []

video = cv2.VideoCapture(0)

while True:

    check, frame = video.read()

    movimento = 0

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if fundo_estatico is None:
        fundo_estatico = cinza
        continue

    #A funcao absdiff pega a diferença absoluta entre pixels, e extrai os pixels dos objetos que estao se movendo
    frame_diferenca = cv2.absdiff(fundo_estatico, cinza)


    # 1[Imagem de origem], 2[Limite de valor de pixels], 3[Valor se o pixel for maior] e 4[Tipo de Threshold]
    frame_limite = cv2.threshold(frame_diferenca, 30, 255, cv2.THRESH_BINARY)[1]

    # Operacao morfologica de dilatação com duas iterações
    frame_limite = cv2.dilate(frame_limite, None, iterations=2)

    cnts, _ = cv2.findContours(frame_limite.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        movimento = 1

        (coordenada_x, coordenada_y, largura, altura) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (coordenada_x, coordenada_y), (coordenada_x + largura, coordenada_y + altura), (0, 0, 0), 3)

    cv2.imshow("Grayscale", cinza)

    cv2.imshow("Frame de diferenca", frame_diferenca)

    cv2.imshow("Frame de Limite", frame_limite)

    cv2.imshow("Frame Colorido com contorno", frame)

    key = cv2.waitKey(1)
    # Tecla S para sair
    if key == ord('s'):

        break

video.release()

cv2.destroyAllWindows()