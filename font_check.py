import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

print("Noto found:", any("Noto" in f.name for f in fm.fontManager.ttflist))