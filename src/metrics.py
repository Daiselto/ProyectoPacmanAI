import csv
import os
from datetime import datetime


class Metrics:
    def __init__(self):
        self.capturas = 0
        self.capturas_por_fantasma = {0: 0, 1: 0, 2: 0, 3: 0}
        self.veces_vulnerable = 0
        self.fantasmas_comidos = 0
        self.frames_totales = 0
        self.fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.nombres = {
            0: "Blinky (Rojo)",
            1: "Pinky (Rosa)",
            2: "Inky (Celeste)",
            3: "Clyde (Naranja)"
        }
        self.CSV_PATH = "metricas_partidas.csv"
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.CSV_PATH):
            with open(self.CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "fecha", "capturas_totales", "power_pellets",
                    "fantasmas_comidos", "duracion_frames",
                    "capturas_blinky", "capturas_pinky",
                    "capturas_inky", "capturas_clyde"
                ])

    def registrar_captura(self, ghost_id: int):
        self.capturas += 1
        self.capturas_por_fantasma[ghost_id] += 1

    def registrar_vulnerable(self):
        self.veces_vulnerable += 1

    def registrar_fantasma_comido(self):
        self.fantasmas_comidos += 1

    def tick(self):
        self.frames_totales += 1

    def guardar_csv(self):
        with open(self.CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.fecha,
                self.capturas,
                self.veces_vulnerable,
                self.fantasmas_comidos,
                self.frames_totales,
                self.capturas_por_fantasma[0],
                self.capturas_por_fantasma[1],
                self.capturas_por_fantasma[2],
                self.capturas_por_fantasma[3]
            ])

    def reporte(self):
        print("\n===== MÉTRICAS DE LA PARTIDA =====")
        print(f"Capturas totales de Pac-Man: {self.capturas}")
        print(f"Veces que Pac-Man comió power pellet: {self.veces_vulnerable}")
        print(f"Fantasmas comidos por Pac-Man: {self.fantasmas_comidos}")
        print(f"Duración de la partida (frames): {self.frames_totales}")
        print("\nCapturas por fantasma:")
        for ghost_id, count in self.capturas_por_fantasma.items():
            print(f"  {self.nombres[ghost_id]}: {count} captura(s)")
        print("==================================\n")
        self.guardar_csv()
        print(f"Métricas guardadas en {self.CSV_PATH}")