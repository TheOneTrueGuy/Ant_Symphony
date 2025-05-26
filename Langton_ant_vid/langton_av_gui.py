import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QSpinBox, QColorDialog, QLineEdit, QFileDialog, QGridLayout, QGroupBox, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPixmap, QFont
import numpy as np
from scipy.interpolate import interp1d
import pygame
import wave
from langton_sim import LangtonAntSimulation

# Initialize Pygame for audio
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)  # Set to stereo

# Audio constants
SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = int(SAMPLE_RATE / 60)  # 60 fps

# Musical scale definitions (frequencies in Hz)
A4 = 440.0  # Reference note
SCALES = {
    'Pentatonic': [0, 2, 4, 7, 9],  # Major pentatonic scale intervals
    'Major': [0, 2, 4, 5, 7, 9, 11],  # Major scale intervals
    'Minor': [0, 2, 3, 5, 7, 8, 10],  # Natural minor scale intervals
    'Chromatic': list(range(12))  # All notes
}

def get_note_frequency(note_number, scale_type='Pentatonic', base_freq=A4):
    """Convert note number to frequency based on selected scale"""
    scale = SCALES[scale_type]
    octave = note_number // len(scale)
    note_index = note_number % len(scale)
    semitones = scale[note_index] + (octave * 12)
    return base_freq * (2 ** (semitones / 12))

def generate_waveform(t, freq, waveform_type='sine'):
    """Generate different types of waveforms"""
    if waveform_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif waveform_type == 'square':
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform_type == 'sawtooth':
        return 2 * (t * freq - np.floor(0.5 + t * freq))
    elif waveform_type == 'triangle':
        return 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1
    return np.sin(2 * np.pi * freq * t)  # Default to sine

def apply_adsr_envelope(signal, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
    """Apply ADSR envelope to the signal"""
    length = len(signal)
    envelope = np.ones(length)
    
    # Calculate sample counts for each stage (using frame size instead of sample rate)
    attack_samples = int(attack * length)
    decay_samples = int(decay * length)
    release_samples = int(release * length)
    
    # Attack
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    if decay_samples > 0:
        decay_start = attack_samples
        decay_end = decay_start + decay_samples
        if decay_end <= length:  # Make sure we don't exceed the signal length
            envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_samples)
    
    # Sustain is already set to sustain level
    
    # Release
    if release_samples > 0:
        release_start = length - release_samples
        if release_start >= 0:  # Make sure we don't go negative
            envelope[release_start:] = np.linspace(sustain, 0, release_samples)
    
    return signal * envelope

def generate_audio_segment(ants, freq_scale, amp_scale, grid_width, interp_points, scale_type='Pentatonic', waveform_type='sine', samples=SAMPLES_PER_FRAME):
    # Create stereo output (2 channels)
    segment = np.zeros((samples, 2))
    
    # Generate audio based on ant positions only
    for ant in ants:
        # Map ant's x position to musical note
        note_number = int((ant.x / grid_width) * 24)  # 2 octaves of notes
        freq = get_note_frequency(note_number, scale_type)
        
        # Map ant's y position to amplitude
        amp = amp_scale[ant.y]
        
        # Calculate pan position (0 = left, 1 = right)
        pan = ant.x / grid_width
        
        # Generate time points for interpolation
        t = np.linspace(0, samples / SAMPLE_RATE, samples)
        
        # Generate the base signal with selected waveform
        signal = amp * generate_waveform(t, freq, waveform_type)
        
        # Apply ADSR envelope
        signal = apply_adsr_envelope(signal)
        
        # Apply panning (left and right channels)
        segment[:, 0] += signal * (1 - pan)  # Left channel
        segment[:, 1] += signal * pan        # Right channel
    
    # Normalize to prevent clipping
    if np.max(np.abs(segment)) > 0:
        segment /= np.max(np.abs(segment))
    
    # Convert to 16-bit integer
    audio_16bit = np.int16(segment * 32767)
    return audio_16bit

class AntPropertyWidget(QWidget):
    def __init__(self, ant_index, color, parent=None):
        super().__init__(parent)
        self.ant_index = ant_index
        self.color = color
        layout = QHBoxLayout()
        font = QFont()
        font.setPointSize(10)
        font.setFamily('Arial')
        # Rule edit
        self.rule_edit = QLineEdit('RL')
        self.rule_edit.setFont(font)
        self.rule_edit.setFixedWidth(40)
        # Step size spin
        self.step_size_spin = QSpinBox()
        self.step_size_spin.setRange(1, 20)
        self.step_size_spin.setValue(1)
        self.step_size_spin.setFont(font)
        self.step_size_spin.setFixedWidth(50)
        # Color button
        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet(f'background-color: {color.name()};')
        self.color_btn.setFixedWidth(30)
        self.color_btn.setFixedHeight(22)
        self.color_btn.setFont(font)
        self.color_btn.clicked.connect(self.pick_color)
        # Labels
        label_rule = QLabel(f"Ant {ant_index+1} Rule:")
        label_rule.setFont(font)
        label_step = QLabel("Step Size:")
        label_step.setFont(font)
        label_color = QLabel("Color:")
        label_color.setFont(font)
        # Add to layout
        layout.addWidget(label_rule)
        layout.addWidget(self.rule_edit)
        layout.addWidget(label_step)
        layout.addWidget(self.step_size_spin)
        layout.addWidget(label_color)
        layout.addWidget(self.color_btn)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setMaximumHeight(32)

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color
            self.color_btn.setStyleSheet(f'background-color: {color.name()};')

    def get_rule(self):
        return self.rule_edit.text()

    def get_step_size(self):
        return self.step_size_spin.value()

    def get_color(self):
        return self.color

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Langton's Ant A/V Generator")
        main_layout = QHBoxLayout()

        # Left: Main Canvas
        self.canvas = QLabel()
        self.canvas.setMinimumSize(400, 400)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background: white; border: 1px solid black;")
        main_layout.addWidget(self.canvas, stretch=1)

        # Right: Controls
        controls_layout = QVBoxLayout()

        # Add scale and waveform selection
        sound_group = QGroupBox("Sound Settings")
        sound_layout = QGridLayout()
        
        # Scale selection
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALES.keys())
        self.scale_combo.setCurrentText('Pentatonic')
        
        # Waveform selection
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(['sine', 'square', 'sawtooth', 'triangle'])
        self.waveform_combo.setCurrentText('sine')
        
        sound_layout.addWidget(QLabel("Scale:"), 0, 0)
        sound_layout.addWidget(self.scale_combo, 0, 1)
        sound_layout.addWidget(QLabel("Waveform:"), 1, 0)
        sound_layout.addWidget(self.waveform_combo, 1, 1)
        sound_group.setLayout(sound_layout)
        controls_layout.addWidget(sound_group)

        # Parameter dropdowns for X and Y
        param_group = QGroupBox("Parameter Mapping")
        param_layout = QGridLayout()
        self.x_param_combo = QComboBox()
        self.x_param_combo.addItems(["Frequency", "Amplitude"])
        self.y_param_combo = QComboBox()
        self.y_param_combo.addItems(["Amplitude", "Frequency"])
        param_layout.addWidget(QLabel("X Axis:"), 0, 0)
        param_layout.addWidget(self.x_param_combo, 0, 1)
        param_layout.addWidget(QLabel("Y Axis:"), 1, 0)
        param_layout.addWidget(self.y_param_combo, 1, 1)
        param_group.setLayout(param_layout)
        controls_layout.addWidget(param_group)

        # Range sliders for X and Y
        range_group = QGroupBox("Parameter Ranges")
        range_layout = QGridLayout()
        self.x_min_slider = QSlider(Qt.Horizontal)
        self.x_min_slider.setRange(20, 2000)
        self.x_min_slider.setValue(200)
        self.x_max_slider = QSlider(Qt.Horizontal)
        self.x_max_slider.setRange(20, 2000)
        self.x_max_slider.setValue(2000)
        self.y_min_slider = QSlider(Qt.Horizontal)
        self.y_min_slider.setRange(1, 100)
        self.y_min_slider.setValue(10)
        self.y_max_slider = QSlider(Qt.Horizontal)
        self.y_max_slider.setRange(1, 100)
        self.y_max_slider.setValue(50)
        
        # Add interpolation points slider
        self.interp_slider = QSlider(Qt.Horizontal)
        self.interp_slider.setRange(2, 50)
        self.interp_slider.setValue(10)
        self.interp_slider.setTickPosition(QSlider.TicksBelow)
        self.interp_slider.setTickInterval(8)
        
        range_layout.addWidget(QLabel("X Min:"), 0, 0)
        range_layout.addWidget(self.x_min_slider, 0, 1)
        range_layout.addWidget(QLabel("X Max:"), 1, 0)
        range_layout.addWidget(self.x_max_slider, 1, 1)
        range_layout.addWidget(QLabel("Y Min:"), 2, 0)
        range_layout.addWidget(self.y_min_slider, 2, 1)
        range_layout.addWidget(QLabel("Y Max:"), 3, 0)
        range_layout.addWidget(self.y_max_slider, 3, 1)
        range_layout.addWidget(QLabel("Smoothness:"), 4, 0)
        range_layout.addWidget(self.interp_slider, 4, 1)
        range_layout.addWidget(QLabel("Delay (ms):"), 5, 0)
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(10, 500)
        self.delay_slider.setValue(50)
        self.delay_slider.setTickPosition(QSlider.TicksBelow)
        self.delay_slider.setTickInterval(50)
        range_layout.addWidget(self.delay_slider, 5, 1)
        range_group.setLayout(range_layout)
        controls_layout.addWidget(range_group)

        # Ant count and per-ant properties
        ant_group = QGroupBox("Ants")
        ant_layout = QVBoxLayout()
        self.ant_count_spin = QSpinBox()
        self.ant_count_spin.setRange(1, 6)
        self.ant_count_spin.setValue(2)
        self.ant_count_spin.valueChanged.connect(self.update_ant_properties)
        ant_layout.addWidget(QLabel("Ant Count:"))
        ant_layout.addWidget(self.ant_count_spin)
        # Scroll area for ant properties
        self.ant_properties_scroll = QScrollArea()
        self.ant_properties_scroll.setWidgetResizable(True)
        self.ant_properties_container = QWidget()
        self.ants_properties_layout = QVBoxLayout()
        self.ant_properties_container.setLayout(self.ants_properties_layout)
        self.ant_properties_scroll.setWidget(self.ant_properties_container)
        ant_layout.addWidget(self.ant_properties_scroll)
        ant_group.setLayout(ant_layout)
        controls_layout.addWidget(ant_group)
        self.ant_properties_widgets = []

        # Iteration limit and controls
        iter_group = QGroupBox("Run Controls")
        iter_layout = QHBoxLayout()
        self.iter_limit_spin = QSpinBox()
        self.iter_limit_spin.setRange(1, 100000)
        self.iter_limit_spin.setValue(1000)
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.quit_btn = QPushButton("Quit")
        iter_layout.addWidget(QLabel("Iteration Limit:"))
        iter_layout.addWidget(self.iter_limit_spin)
        iter_layout.addWidget(self.pause_btn)
        iter_layout.addWidget(self.resume_btn)
        iter_layout.addWidget(self.quit_btn)
        iter_group.setLayout(iter_layout)
        controls_layout.addWidget(iter_group)

        # Output controls
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        self.save_audio_btn = QPushButton("Save Audio")
        self.save_video_btn = QPushButton("Save Video")
        self.combine_btn = QPushButton("Combine A/V (ffmpeg)")
        output_layout.addWidget(self.save_audio_btn)
        output_layout.addWidget(self.save_video_btn)
        output_layout.addWidget(self.combine_btn)
        output_group.setLayout(output_layout)
        controls_layout.addWidget(output_group)

        # Connect output buttons
        self.save_audio_btn.clicked.connect(self.save_audio)
        self.save_video_btn.clicked.connect(self.save_video)
        self.combine_btn.clicked.connect(self.combine_av)

        # GO! button
        self.go_btn = QPushButton("GO!")
        controls_layout.addWidget(self.go_btn)

        # Status label
        self.status_label = QLabel("")
        controls_layout.addWidget(self.status_label)

        main_layout.addLayout(controls_layout, stretch=0)
        self.setLayout(main_layout)

        # Initialize ant properties
        self.update_ant_properties()

        # Timer for real-time simulation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.simulation_step)
        self.sim = None
        self.iter_limit = 0

        # Connect buttons
        self.go_btn.clicked.connect(self.start_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.resume_btn.clicked.connect(self.resume_simulation)
        self.quit_btn.clicked.connect(self.stop_simulation)

    def update_ant_properties(self):
        for w in self.ant_properties_widgets:
            self.ants_properties_layout.removeWidget(w)
            w.deleteLater()
        self.ant_properties_widgets = []
        for i in range(self.ant_count_spin.value()):
            color = QColor([255, 0, 0, 255][i % 4], [0, 255, 0, 255][i % 4], [0, 0, 255, 255][i % 4])
            w = AntPropertyWidget(i, color)
            self.ant_properties_widgets.append(w)
            self.ants_properties_layout.addWidget(w)
        # Set a max height for the scroll area (e.g., enough for 3 ants before scrolling)
        self.ant_properties_scroll.setMaximumHeight(110)

    def collect_simulation_config(self):
        width, height, cell_size = 800, 800, 5
        ant_count = self.ant_count_spin.value()
        grid_width = width // cell_size
        grid_height = height // cell_size
        ant_configs = []
        center_x = grid_width // 2
        center_y = grid_height // 2
        spacing = 5
        for i, w in enumerate(self.ant_properties_widgets):
            offset = i * spacing
            ant_cfg = {
                'x': (center_x + offset) % grid_width,
                'y': (center_y + offset) % grid_height,
                'color': w.get_color().getRgb()[:3],
                'rule': w.get_rule(),
                'step_size': w.get_step_size(),
            }
            ant_configs.append(ant_cfg)
        return width, height, cell_size, ant_configs

    def start_simulation(self):
        width, height, cell_size, ant_configs = self.collect_simulation_config()
        self.sim = LangtonAntSimulation(width, height, cell_size, ant_configs)
        self.iter_limit = self.iter_limit_spin.value()
        self.status_label.setText('Simulation running...')
        # Use the delay slider value for the timer interval
        self.timer.start(self.delay_slider.value())
        # Clear any stored paused state
        if hasattr(self, 'paused_state'):
            delattr(self, 'paused_state')

    def simulation_step(self):
        if self.sim and self.sim.get_iteration() < self.iter_limit:
            self.sim.step()
            self.update_canvas()
            
            # Generate and play audio with logarithmic scaling
            min_amp = self.y_min_slider.value() / 100
            max_amp = self.y_max_slider.value() / 100
            amp_scale = np.exp(np.linspace(np.log(max(min_amp, 0.001)), np.log(max_amp), self.sim.grid_height))
            
            # Get current interpolation points from slider
            interp_points = self.interp_slider.value()
            
            # Get current scale and waveform
            scale_type = self.scale_combo.currentText()
            waveform_type = self.waveform_combo.currentText()
            
            # Calculate samples based on the delay
            delay_ms = self.delay_slider.value()
            samples = int((delay_ms / 1000.0) * SAMPLE_RATE)
            
            audio_segment = generate_audio_segment(
                self.sim.get_ants(), 
                None,
                amp_scale, 
                self.sim.grid_width, 
                interp_points,
                scale_type,
                waveform_type,
                samples
            )
            try:
                sound = pygame.sndarray.make_sound(audio_segment)
                sound.play()
            except Exception as e:
                print(f"Audio error: {e}")
        else:
            self.timer.stop()
            self.status_label.setText("Simulation complete")

    def update_canvas(self):
        # Use the current canvas size
        width = self.canvas.width()
        height = self.canvas.height()
        cell_size = max(1, min(width, height) // (self.sim.grid_width if self.sim else 160))
        grid = self.sim.get_grid()
        ants = self.sim.get_ants()
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        
        # Draw grid cells with ant-specific colors
        # TODO: In the future, we could make ants interact differently with different colored trails
        # For example:
        # - Ants could follow trails of their own color
        # - Ants could avoid trails of other colors
        # - Different colored trails could have different rules (e.g., some colors could be "walls")
        # - Trails could fade over time or have different persistence rules
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                state, ant_index = grid[x, y]
                if state != 0:  # If the cell has been visited
                    # The ant_index is 1-based, so subtract 1 to get the actual ant
                    if 0 < ant_index <= len(ants):
                        # Use the ant's color for its trail, but make it slightly transparent
                        ant_color = QColor(*ants[ant_index-1].color)
                        ant_color.setAlpha(128)  # 50% transparency
                        painter.fillRect(x * cell_size, y * cell_size, cell_size, cell_size, ant_color)
        
        # Draw ants
        for ant in ants:
            color = QColor(*ant.color)
            painter.fillRect(ant.x * cell_size, ant.y * cell_size, cell_size, cell_size, color)
        painter.end()
        self.canvas.setPixmap(pixmap)

    def pause_simulation(self):
        self.timer.stop()
        self.status_label.setText('Simulation paused.')
        # Store current state for potential resume
        if self.sim:
            self.paused_state = {
                'grid': self.sim.grid.copy(),
                'ants': [(ant.x, ant.y, ant.direction) for ant in self.sim.ants],
                'iteration': self.sim.iteration
            }

    def resume_simulation(self):
        if self.sim is not None:
            # Reset iteration count to 0 but keep current state
            self.sim.reset_iteration_count()
            # Use the delay slider value for the timer interval
            self.timer.start(self.delay_slider.value())
            self.status_label.setText('Simulation running...')

    def stop_simulation(self):
        self.timer.stop()
        self.sim = None
        self.status_label.setText('Simulation stopped.')
        self.canvas.clear()
        # Clear any stored paused state
        if hasattr(self, 'paused_state'):
            delattr(self, 'paused_state')

    def save_audio(self):
        if not self.sim:
            self.status_label.setText("No simulation running to save")
            return

        # Get save location from user
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", "WAV Files (*.wav);;All Files (*)"
        )
        if not filename:
            return

        try:
            # Get current parameters
            min_amp = self.y_min_slider.value() / 100
            max_amp = self.y_max_slider.value() / 100
            amp_scale = np.exp(np.linspace(np.log(max(min_amp, 0.001)), np.log(max_amp), self.sim.grid_height))
            interp_points = self.interp_slider.value()
            scale_type = self.scale_combo.currentText()
            waveform_type = self.waveform_combo.currentText()
            
            # Calculate total samples needed
            total_steps = self.iter_limit_spin.value()
            samples_per_step = int((self.delay_slider.value() / 1000.0) * SAMPLE_RATE)
            total_samples = total_steps * samples_per_step
            
            # Pre-allocate the full audio array
            full_audio = np.zeros((total_samples, 2), dtype=np.int16)
            
            # Create a copy of the simulation
            sim_copy = LangtonAntSimulation(
                self.sim.width, self.sim.height, self.sim.cell_size,
                [{'x': ant.x, 'y': ant.y, 'color': ant.color, 'rule': ant.rule, 'step_size': ant.step_size}
                 for ant in self.sim.ants]
            )
            
            # Generate audio for each step
            for step in range(total_steps):
                # Generate audio segment for current state
                audio_segment = generate_audio_segment(
                    sim_copy.get_ants(),
                    None,
                    amp_scale,
                    sim_copy.grid_width,
                    interp_points,
                    scale_type,
                    waveform_type,
                    samples_per_step
                )
                
                # Place the segment in the correct position
                start_idx = step * samples_per_step
                end_idx = start_idx + samples_per_step
                if end_idx <= total_samples:  # Make sure we don't exceed array bounds
                    full_audio[start_idx:end_idx] = audio_segment
                
                # Step the simulation
                sim_copy.step()
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(full_audio.tobytes())
            
            self.status_label.setText(f"Audio saved to {filename}")
        except Exception as e:
            self.status_label.setText(f"Error saving audio: {str(e)}")

    def save_video(self):
        if not self.sim:
            self.status_label.setText("No simulation running to save")
            return

        # Get save location from user
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "MP4 Files (*.mp4);;All Files (*)"
        )
        if not filename:
            return

        try:
            import cv2
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 60.0, (800, 800))
            
            # Create a copy of the simulation
            sim_copy = LangtonAntSimulation(
                self.sim.width, self.sim.height, self.sim.cell_size,
                [{'x': ant.x, 'y': ant.y, 'color': ant.color, 'rule': ant.rule, 'step_size': ant.step_size}
                 for ant in self.sim.ants]
            )
            
            # Generate frames
            for _ in range(self.iter_limit_spin.value()):
                # Create frame
                frame = np.zeros((800, 800, 3), dtype=np.uint8)
                frame.fill(255)  # White background
                
                # Draw grid
                grid = sim_copy.get_grid()
                ants = sim_copy.get_ants()
                for x in range(grid.shape[0]):
                    for y in range(grid.shape[1]):
                        state, ant_index = grid[x, y]
                        if state != 0:
                            if 0 < ant_index <= len(ants):
                                color = ants[ant_index-1].color
                                frame[y*5:(y+1)*5, x*5:(x+1)*5] = color
                
                # Draw ants
                for ant in ants:
                    frame[ant.y*5:(ant.y+1)*5, ant.x*5:(ant.x+1)*5] = ant.color
                
                # Write frame
                out.write(frame)
                sim_copy.step()
            
            out.release()
            self.status_label.setText(f"Video saved to {filename}")
        except Exception as e:
            self.status_label.setText(f"Error saving video: {str(e)}")

    def combine_av(self):
        if not self.sim:
            self.status_label.setText("No simulation running to combine")
            return

        # Get save location from user
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Combined A/V", "", "MP4 Files (*.mp4);;All Files (*)"
        )
        if not filename:
            return

        try:
            # First save audio and video separately
            import tempfile
            import os
            import subprocess
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save audio
                audio_file = os.path.join(temp_dir, "temp_audio.wav")
                self.save_audio()
                
                # Save video
                video_file = os.path.join(temp_dir, "temp_video.mp4")
                self.save_video_to_file(video_file)
                
                # Combine using ffmpeg
                subprocess.run([
                    'ffmpeg', '-i', video_file, '-i', audio_file,
                    '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                    filename
                ], check=True)
            
            self.status_label.setText(f"Combined A/V saved to {filename}")
        except Exception as e:
            self.status_label.setText(f"Error combining A/V: {str(e)}")

    def save_video_to_file(self, filename):
        # Similar to save_video but without the file dialog
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 60.0, (800, 800))
        
        sim_copy = LangtonAntSimulation(
            self.sim.width, self.sim.height, self.sim.cell_size,
            [{'x': ant.x, 'y': ant.y, 'color': ant.color, 'rule': ant.rule, 'step_size': ant.step_size}
             for ant in self.sim.ants]
        )
        
        for _ in range(self.iter_limit_spin.value()):
            frame = np.zeros((800, 800, 3), dtype=np.uint8)
            frame.fill(255)
            
            grid = sim_copy.get_grid()
            ants = sim_copy.get_ants()
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    state, ant_index = grid[x, y]
                    if state != 0:
                        if 0 < ant_index <= len(ants):
                            color = ants[ant_index-1].color
                            frame[y*5:(y+1)*5, x*5:(x+1)*5] = color
            
            for ant in ants:
                frame[ant.y*5:(ant.y+1)*5, ant.x*5:(ant.x+1)*5] = ant.color
            
            out.write(frame)
            sim_copy.step()
        
        out.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 