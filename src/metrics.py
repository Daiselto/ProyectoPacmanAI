class Metrics:
    def __init__(self):
        self.capturas = 0
        self.capturas_por_fantasma = {0: 0, 1: 0, 2: 0, 3: 0}
        self.veces_vulnerable = 0
        self.fantasmas_comidos = 0
        self.frames_totales = 0
        self.nombres = {
            0: "Blinky (Rojo)",
            1: "Pinky (Rosa)",
            2: "Inky (Celeste)",
            3: "Clyde (Naranja)"
        }

    def registrar_captura(self, ghost_id: int):
        self.capturas += 1
        self.capturas_por_fantasma[ghost_id] += 1

    def registrar_vulnerable(self):
        self.veces_vulnerable += 1

    def registrar_fantasma_comido(self):
        self.fantasmas_comidos += 1

    def tick(self):
        self.frames_totales += 1

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