# GO1310-Cv2Ultralytics
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class EPIDetector:
    def __init__(self, model_path='yolov8m.pt'):
        self.model = YOLO(model_path)
        self.conformidade = defaultdict(int)
        self.violacoes = defaultdict(int)

    def detect_frame(self, frame):
        """Detecta EPIs em um frame"""
        results = self.model.predict(frame, conf=0.5, verbose=False)

        pessoas = []
        capacetes = []
        coletes = []

        # Extrair detecções
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            bbox = [x1, y1, x2, y2]

            if cls == 0:  # pessoa
                pessoas.append(bbox)
            elif cls == 1:  # capacete
                capacetes.append(bbox)
            elif cls == 2:  # colete
                coletes.append(bbox)

        # Verificar conformidade
        self.verificar_conformidade(pessoas, capacetes, coletes)

        # Anotar frame
        annotated = self.anotar_frame(frame, pessoas, capacetes, coletes)

        return annotated

    def verificar_conformidade(self, pessoas, capacetes, coletes):
        """Verifica se pessoas estão com EPIs"""
        for pessoa in pessoas:
            tem_capacete = self.tem_epi_proximo(pessoa, capacetes)
            tem_colete = self.tem_epi_proximo(pessoa, coletes)

            if tem_capacete and tem_colete:
                self.conformidade['completo'] += 1
            elif tem_capacete or tem_colete:
                self.conformidade['parcial'] += 1
                self.violacoes['falta_epi'] += 1
            else:
                self.violacoes['sem_epi'] += 1

    def tem_epi_proximo(self, pessoa_box, epi_boxes, threshold=50):
        """Verifica se EPI está próximo da pessoa"""
        px1, py1, px2, py2 = pessoa_box
        pessoa_centro = ((px1+px2)/2, (py1+py2)/2)

        for epi_box in epi_boxes:
            ex1, ey1, ex2, ey2 = epi_box
            epi_centro = ((ex1+ex2)/2, (ey1+ey2)/2)

            # Distância Euclidiana
            dist = np.sqrt((pessoa_centro[0]-epi_centro[0])**2 + 
                          (pessoa_centro[1]-epi_centro[1])**2)

            if dist < threshold:
                return True
        return False

    def anotar_frame(self, frame, pessoas, capacetes, coletes):
        """Adiciona anotações visuais"""
        annotated = frame.copy()

        # Desenhar pessoas (vermelho se sem EPI)
        for pessoa in pessoas:
            x1, y1, x2, y2 = map(int, pessoa)
            tem_capacete = self.tem_epi_proximo(pessoa, capacetes)
            tem_colete = self.tem_epi_proximo(pessoa, coletes)

            if tem_capacete and tem_colete:
                cor = (0, 255, 0)  # Verde - conforme
                texto = "OK"
            else:
                cor = (0, 0, 255)  # Vermelho - não conforme
                texto = "ALERTA"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(annotated, texto, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        # Estatísticas
        stats = f"Conforme: {self.conformidade['completo']} | " \
                f"Violacoes: {self.violacoes['sem_epi']}"
        cv2.putText(annotated, stats, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return annotated

    def processar_video(self, video_path, output_path='output.mp4'):
        """Processa vídeo completo"""
        cap = cv2.VideoCapture(video_path)

        # Configurar output
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.detect_frame(frame)
            out.write(annotated)

            cv2.imshow('EPIs', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Relatório final
        self.gerar_relatorio()

    def gerar_relatorio(self):
        """Gera relatório de conformidade"""
        total = sum(self.conformidade.values())
        conforme_pct = (self.conformidade['completo'] / total * 100) if total > 0 else 0

        print("\n" + "="*50)
        print("RELATÓRIO DE CONFORMIDADE DE EPIs")
        print("="*50)
        print(f"Total de detecções: {total}")
        print(f"Conformes: {self.conformidade['completo']} ({conforme_pct:.1f}%)")
        print(f"Parcialmente: {self.conformidade['parcial']}")
        print(f"Violações totais: {sum(self.violacoes.values())}")
        print("="*50)

# USO:


if __name__ == "__main__":
    detector = EPIDetector('yolov8_epi_custom.pt')
    detector.processar_video('obra_video.mp4', 'output_epi.mp4')
