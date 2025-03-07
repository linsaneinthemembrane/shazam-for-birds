# bird_song_id/audio.py
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class AudioProcessor:
    def __init__(self):
        """Initialize the audio processor for bird song identification"""
        self.sample_rate = None
        self.audio_data = None
        
    def load_audio(self, file_path, target_sr=22050):
        """
        Load an audio file and store its data and sample rate
        Optionally downsample to reduce processing overhead
        
        Parameters:
        - file_path: Path to the audio file
        - target_sr: Target sample rate for downsampling (22.05kHz is sufficient for bird songs)
        
        Returns:
        - audio_data: Processed audio samples
        - sample_rate: Sample rate in Hz
        """
        self.sample_rate, self.audio_data = wavfile.read(file_path)
        print(f"Original audio: {self.audio_data.shape} samples at {self.sample_rate}Hz")
        
        # Convert to mono if stereo
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
            print(f"Converted stereo to mono: {self.audio_data.shape} samples")
        
        # Downsample if needed (most bird songs are under 10kHz, so Nyquist rate of 20kHz is sufficient)
        if self.sample_rate > target_sr:
            # Calculate the number of samples in the resampled audio
            new_length = int(len(self.audio_data) * target_sr / self.sample_rate)
            self.audio_data = signal.resample(self.audio_data, new_length)
            self.sample_rate = target_sr
            print(f"Downsampled to {self.audio_data.shape} samples at {self.sample_rate}Hz")
        
        # Normalize audio amplitude
        self.audio_data = self.audio_data / (np.max(np.abs(self.audio_data)) + 1e-10)
        
        return self.audio_data, self.sample_rate
    
    def create_spectrogram(self, window_size=1024, hop_length=512):
        """
        Generate a spectrogram from the loaded audio data using the sliding window technique
        
        Parameters:
        - window_size: Size of each window segment in samples
        - hop_length: Number of samples to advance between windows (overlap = window_size - hop_length)
        
        Returns:
        - frequencies: Array of frequency bins
        - times: Array of time points
        - spectrogram: 2D array where rows=frequencies, columns=time, values=intensity
        """
        if self.audio_data is None:
            raise ValueError("No audio data loaded. Call load_audio first.")
        
        # Apply sliding window with Hamming window to reduce spectral leakage
        # The window function tapers signal at edges, creating smooth transitions
        frequencies, times, spectrogram = signal.spectrogram(
            self.audio_data, 
            fs=self.sample_rate,
            window='hamming',  # Explicitly use Hamming window to reduce spectral leakage
            nperseg=window_size,  # Window size
            noverlap=window_size - hop_length,  # Overlap between windows
            scaling='spectrum',
            mode='magnitude'  # Return magnitude for better visualization
        )
        
        print(f"Generated spectrogram with shape: {spectrogram.shape} (frequencies × time points)")
        print(f"Frequency range: {frequencies[0]:.1f}Hz to {frequencies[-1]:.1f}Hz")
        
        return frequencies, times, spectrogram
    
    def extract_peaks(self, spectrogram, frequencies, times, num_bands=6, min_freq=500, max_freq=10000):
        """
        Extract significant peaks from a spectrogram using logarithmic frequency bands.
        
        Parameters:
        - spectrogram: 2D array where rows=frequencies, columns=time
        - frequencies: Array of frequency values corresponding to spectrogram rows
        - times: Array of time values corresponding to spectrogram columns
        - num_bands: Number of logarithmic frequency bands to use
        - min_freq: Minimum frequency to consider (Hz)
        - max_freq: Maximum frequency to consider (Hz)
        
        Returns:
        - peaks: List of (time_idx, freq_idx, intensity) tuples representing significant peaks
        """
        # Filter spectrogram to focus on the most relevant frequency range
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        filtered_freqs = frequencies[freq_mask]
        filtered_spec = spectrogram[freq_mask, :]
        
        print(f"Filtering frequencies from {min_freq}Hz to {max_freq}Hz")
        
        # Create logarithmic frequency bands that mimic human auditory perception
        min_log = np.log10(min_freq)
        max_log = np.log10(max_freq)
        log_bands = np.logspace(min_log, max_log, num_bands + 1)
        
        print(f"Created {num_bands} logarithmic frequency bands")
        print(f"Band boundaries: {', '.join([f'{f:.1f}Hz' for f in log_bands])}")
        
        # Initialize array to store peaks
        all_peaks = []
        
        # Process each time frame
        for t_idx in range(filtered_spec.shape[1]):
            # Get spectrum at this time point
            spectrum = filtered_spec[:, t_idx]
            band_peaks = []
            
            # Find strongest peak in each frequency band
            for band_idx in range(num_bands):
                band_min = log_bands[band_idx]
                band_max = log_bands[band_idx + 1]
                
                # Find indices of frequencies in this band
                band_mask = (filtered_freqs >= band_min) & (filtered_freqs < band_max)
                if not np.any(band_mask):
                    continue
                    
                # Get spectrum values in this band
                band_spectrum = spectrum[band_mask]
                band_freqs_idx = np.where(band_mask)[0]
                
                # Find the strongest peak in this band
                if len(band_spectrum) > 0:
                    max_idx = np.argmax(band_spectrum)
                    # Calculate original frequency index in the unfiltered spectrogram
                    orig_freq_idx = np.where(freq_mask)[0][0] + band_freqs_idx[max_idx]
                    magnitude = band_spectrum[max_idx]
                    band_peaks.append((t_idx, orig_freq_idx, magnitude))
            
            # Calculate dynamic threshold using the average magnitude
            if band_peaks:
                magnitudes = [p[2] for p in band_peaks]
                threshold = np.mean(magnitudes)
                
                # Keep only peaks above threshold
                significant_peaks = [p for p in band_peaks if p[2] >= threshold]
                all_peaks.extend(significant_peaks)
        
        print(f"Extracted {len(all_peaks)} significant peaks from spectrogram")
        return all_peaks
    
    def create_fingerprint_from_peaks(self, peaks, frequencies, times, fan_out=5, target_zone_seconds=3.0):
        """
        Create a robust audio fingerprint using anchor-target peak pairs
        
        Parameters:
        - peaks: List of (time_idx, freq_idx, intensity) tuples
        - frequencies: Array of frequency values
        - times: Array of time points
        - fan_out: Number of target points to pair with each anchor
        - target_zone_seconds: Time range ahead of anchor to look for targets
        
        Returns:
        - fingerprint: List of (hash_value, anchor_time) tuples
        """
        # Sort peaks by time to ensure proper target zone selection
        peaks.sort(key=lambda x: x[0])  # Sort by time index
        
        # Convert from time indices to actual seconds
        peak_times = [times[p[0]] for p in peaks]
        peak_freqs = [frequencies[p[1]] for p in peaks]
        
        print(f"Creating fingerprint from {len(peaks)} peaks")
        fingerprint = []
        
        # Process each peak as an anchor point
        for i, anchor in enumerate(peaks):
            anchor_time_idx, anchor_freq_idx, _ = anchor
            anchor_time = times[anchor_time_idx]
            anchor_freq = frequencies[anchor_freq_idx]
            
            # Define the target zone boundary
            target_zone_end = anchor_time + target_zone_seconds
            
            # Find target peaks within the target zone
            targets = []
            j = i + 1  # Start from the next peak
            
            # Collect potential targets
            while j < len(peaks) and peak_times[j] <= target_zone_end:
                targets.append((j, peaks[j]))
                j += 1
                
                # Limit number of targets to prevent excessive fingerprints
                if len(targets) >= fan_out * 3:  # Collect more than needed to select best ones
                    break
            
            # If we found enough targets, select a subset based on intensity
            if len(targets) >= fan_out:
                # Sort targets by intensity (strongest first) and take the top ones
                targets.sort(key=lambda x: x[1][2], reverse=True)
                targets = targets[:fan_out]
                
                # Create fingerprints for each anchor-target pair
                for target_idx, target in targets:
                    target_time = peak_times[target_idx]
                    target_freq = peak_freqs[target_idx]
                    
                    # Calculate time delta between anchor and target
                    time_delta = target_time - anchor_time
                    
                    # Create hash from anchor freq, target freq, and time delta
                    # We'll quantize the values to make matching more robust
                    anchor_freq_bin = int(anchor_freq)
                    target_freq_bin = int(target_freq)
                    delta_bin = int(time_delta * 10)  # Store with 0.1s precision
                    
                    # Combine into a 32-bit integer hash
                    # Note: We're using bit shifting to pack the values
                    # This is a common technique in audio fingerprinting
                    hash_value = (
                        (anchor_freq_bin & 0x3FF) << 20 |  # 10 bits for anchor frequency
                        (target_freq_bin & 0x3FF) << 10 |  # 10 bits for target frequency
                        (delta_bin & 0x3FF)                # 10 bits for time delta
                    )
                    
                    # Add to fingerprint collection (hash, anchor_time)
                    # The bird_id will be added when storing in the database
                    fingerprint.append((hash_value, anchor_time))
        
        print(f"Generated {len(fingerprint)} fingerprint hashes")
        return fingerprint
    
    def plot_spectrogram(self, frequencies, times, spectrogram, title="Bird Song Spectrogram"):
        """Plot a spectrogram for visualization with logarithmic intensity scale"""
        plt.figure(figsize=(12, 6))
        
        # Use log scale for better visualization of intensity variations
        # Adding a small constant to avoid log(0)
        log_spectrogram = 10 * np.log10(spectrogram + 1e-9)
        
        plt.pcolormesh(times, frequencies, log_spectrogram, shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(title)
        plt.colorbar(label='Intensity [dB]')
        
        # Set y-axis to log scale which is better for audio analysis
        plt.yscale('log')
        plt.ylim(frequencies[0], frequencies[-1])
        
        # Add gridlines for better readability
        plt.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_peaks(self, frequencies, times, spectrogram, peaks, max_peaks_to_show=200):
        """Visualize the spectrogram with extracted peaks highlighted"""
        plt.figure(figsize=(12, 8))
        
        # Plot spectrogram
        log_spec = 10 * np.log10(spectrogram + 1e-9)  # Convert to dB scale
        plt.pcolormesh(times, frequencies, log_spec, shading='gouraud', cmap='viridis')
        
        # Plot peaks (limit to avoid cluttering the visualization)
        if len(peaks) > max_peaks_to_show:
            # Select a subset of peaks by taking every nth peak
            step = len(peaks) // max_peaks_to_show
            peak_subset = peaks[::step]
        else:
            peak_subset = peaks
        
        # Extract coordinates for plotting
        peak_times = [times[p[0]] for p in peak_subset]
        peak_freqs = [frequencies[p[1]] for p in peak_subset]
        
        plt.scatter(peak_times, peak_freqs, color='red', s=15, alpha=0.7, marker='x')
        
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram with Extracted Peaks')
        plt.colorbar(label='Intensity [dB]')
        plt.yscale('log')  # Log scale for better visualization of frequency distribution
        plt.tight_layout()
        plt.show()
    
    def visualize_fingerprint(self, frequencies, times, spectrogram, fingerprint, peaks):
        """Visualize the fingerprint anchor-target pairs overlaid on the spectrogram"""
        plt.figure(figsize=(14, 8))
        
        # Plot the spectrogram
        log_spec = 10 * np.log10(spectrogram + 1e-9)
        plt.pcolormesh(times, frequencies, log_spec, shading='gouraud', cmap='viridis', alpha=0.7)
        
        # Set up colors for visualization
        num_colors = min(10, len(fingerprint) // fan_out)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
        
        # Extract peaks for easier lookup
        peak_dict = {}
        for i, peak in enumerate(peaks):
            t_idx, f_idx, _ = peak
            peak_dict[(t_idx, f_idx)] = i
        
        # Collect anchor-target pairs
        pairs_to_plot = []
        for hash_value, anchor_time in fingerprint[:100]:  # Limit to 100 pairs for clarity
            # Extract components from the hash
            delta_bin = hash_value & 0x3FF
            target_freq_bin = (hash_value >> 10) & 0x3FF
            anchor_freq_bin = (hash_value >> 20) & 0x3FF
            
            # Find closest matching peaks
            anchor_peak = None
            target_peak = None
            
            # Find the anchor peak
            min_dist = float('inf')
            for peak in peaks:
                t_idx, f_idx, _ = peak
                t = times[t_idx]
                f = frequencies[f_idx]
                
                # Check if this could be our anchor
                if abs(t - anchor_time) < 0.1:  # Within 0.1s
                    dist = abs(f - anchor_freq_bin)
                    if dist < min_dist:
                        min_dist = dist
                        anchor_peak = (t, f)
            
            # If we found an anchor, look for its target
            if anchor_peak:
                time_delta = delta_bin / 10  # Convert back from 0.1s precision
                target_time = anchor_time + time_delta
                
                # Find the target peak
                min_dist = float('inf')
                for peak in peaks:
                    t_idx, f_idx, _ = peak
                    t = times[t_idx]
                    f = frequencies[f_idx]
                    
                    # Check if this could be our target
                    if abs(t - target_time) < 0.1:  # Within 0.1s
                        dist = abs(f - target_freq_bin)
                        if dist < min_dist:
                            min_dist = dist
                            target_peak = (t, f)
                
                if target_peak:
                    pairs_to_plot.append((anchor_peak, target_peak))
        
        # Plot the pairs
        for i, (anchor, target) in enumerate(pairs_to_plot):
            color = colors[i % num_colors]
            
            # Plot anchor and target
            plt.scatter(anchor[0], anchor[1], color=color, s=30, alpha=0.8, marker='o')
            plt.scatter(target[0], target[1], color=color, s=30, alpha=0.8, marker='s')
            
            # Draw line connecting them
            plt.plot([anchor[0], target[0]], [anchor[1], target[1]], color=color, alpha=0.6, linewidth=1)
        
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Fingerprint Visualization: Anchor-Target Pairs')
        plt.colorbar(label='Intensity [dB]')
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
    
    def process_audio_file(self, file_path, plot=True):
        """
        Process an audio file through the complete pipeline:
        load → spectrogram → find peaks → create fingerprint → generate hashes
        
        Parameters:
        - file_path: Path to the audio file
        - plot: Whether to plot the spectrogram with peaks
        
        Returns:
        - fingerprint: List of (hash_value, anchor_time) tuples
        """
        # Load and process audio
        self.load_audio(file_path)
        
        # Create spectrogram
        frequencies, times, spectrogram = self.create_spectrogram()
        
        # Find spectral peaks
        peaks = self.extract_peaks(spectrogram, frequencies, times)
        
        # Create fingerprint from peaks
        fingerprint = self.create_fingerprint_from_peaks(peaks, frequencies, times)
        
        # Plot if requested
        if plot:
            # Show the spectrogram
            self.plot_spectrogram(frequencies, times, spectrogram, title=f"Spectrogram: {os.path.basename(file_path)}")
            
            # Show the peaks
            self.visualize_peaks(frequencies, times, spectrogram, peaks)
            
            # Visualize some fingerprint pairs
            try:
                self.visualize_fingerprint(frequencies, times, spectrogram, fingerprint, peaks)
            except Exception as e:
                print(f"Fingerprint visualization error: {e}")
        
        return fingerprint
