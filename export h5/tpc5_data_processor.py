"""
TPC5 Data Processor
A clean, reusable module for loading and processing TPC5 HDF5 files.

Original TPC5 helper functions: Copyright 2017 Elsys AG
Refactored data processor implementation: Copyright Chun-Yu Ke
All rights reserved.
"""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path


class TPC5DataProcessor:
    """
    A comprehensive data processor for TPC5 files with enhanced functionality.
    
    Features:
    - Load voltage and physical data from multiple channels and blocks
    - Extract metadata and timing information
    - Generate time axes for plotting
    - Batch processing capabilities
    - Error handling and validation
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the TPC5 data processor.
        
        Args:
            filepath: Path to the TPC5 file
        """
        self.filepath = Path(filepath)
        self.file_handle = None
        self._validate_file()
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def _validate_file(self):
        """Validate that the file exists and has the correct extension."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"TPC5 file not found: {self.filepath}")
        if self.filepath.suffix.lower() != '.tpc5':
            logging.warning(f"File {self.filepath} may not be a TPC5 file (expected .tpc5 extension)")
    
    def open(self):
        """Open the TPC5 file for reading."""
        try:
            self.file_handle = h5py.File(str(self.filepath), "r")
            logging.info(f"Opened TPC5 file: {self.filepath}")
        except Exception as e:
            raise IOError(f"Failed to open TPC5 file {self.filepath}: {e}")
    
    def close(self):
        """Close the TPC5 file."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            logging.info(f"Closed TPC5 file: {self.filepath}")
    
    # Core TPC5 helper methods (refactored from original)
    def _get_dataset_name(self, channel: int, block: int = 1) -> str:
        """Generate dataset name for raw data."""
        return f'/measurements/00000001/channels/{channel:08d}/blocks/{block:08d}/raw'
    
    def _get_channel_group_name(self, channel: int) -> str:
        """Generate channel group name."""
        return f'/measurements/00000001/channels/{channel:08d}/'
    
    def _get_block_name(self, channel: int, block: int) -> str:
        """Generate block name."""
        return f'/measurements/00000001/channels/{channel:08d}/blocks/{block:08d}/'
    
    def get_available_channels(self) -> List[int]:
        """
        Get list of available channels in the file.
        
        Returns:
            List of channel numbers
        """
        if self.file_handle is None:
            raise RuntimeError("File not opened. Use open() or context manager.")
            
        channels = []
        try:
            channels_group = self.file_handle['/measurements/00000001/channels/']
            for key in channels_group.keys():
                try:
                    channels.append(int(key))
                except ValueError:
                    continue
            return sorted(channels)
        except KeyError:
            return []
    
    def get_available_blocks(self, channel: int) -> List[int]:
        """
        Get list of available blocks for a specific channel.
        
        Args:
            channel: Channel number
            
        Returns:
            List of block numbers
        """
        if self.file_handle is None:
            raise RuntimeError("File not opened. Use open() or context manager.")
            
        blocks = []
        try:
            blocks_group = self.file_handle[f'/measurements/00000001/channels/{channel:08d}/blocks/']
            for key in blocks_group.keys():
                try:
                    blocks.append(int(key))
                except ValueError:
                    continue
            return sorted(blocks)
        except KeyError:
            return []
    
    def get_voltage_data(self, channel: int, block: int = 1) -> np.ndarray:
        """
        Get voltage data for specified channel and block.
        
        Args:
            channel: Channel number
            block: Block number (default: 1)
            
        Returns:
            Voltage data array
        """
        if self.file_handle is None:
            raise RuntimeError("File not opened. Use open() or context manager.")
            
        try:
            channel_group = self.file_handle[self._get_channel_group_name(channel)]
            dataset_name = self._get_dataset_name(channel, block)
            
            # Get scaling parameters
            bin_to_voltage_factor = channel_group.attrs['binToVoltFactor']
            bin_to_voltage_constant = channel_group.attrs['binToVoltConstant']
            
            # Get analog mask for data separation
            analog_mask = channel_group.attrs['analogMask']
            
            # Extract analog data
            analog_data = self.file_handle[dataset_name] & analog_mask
            
            # Scale to voltage
            return analog_data * bin_to_voltage_factor + bin_to_voltage_constant
            
        except KeyError as e:
            raise ValueError(f"Channel {channel}, block {block} not found in file: {e}")
    
    def get_physical_data(self, channel: int, block: int = 1) -> np.ndarray:
        """
        Get physical data (scaled to engineering units) for specified channel and block.
        
        Args:
            channel: Channel number
            block: Block number (default: 1)
            
        Returns:
            Physical data array in engineering units
        """
        if self.file_handle is None:
            raise RuntimeError("File not opened. Use open() or context manager.")
            
        try:
            channel_group = self.file_handle[self._get_channel_group_name(channel)]
            dataset_name = self._get_dataset_name(channel, block)
            
            # Get scaling parameters
            bin_to_voltage_factor = channel_group.attrs['binToVoltFactor']
            bin_to_voltage_constant = channel_group.attrs['binToVoltConstant']
            volt_to_physical_factor = channel_group.attrs['voltToPhysicalFactor']
            volt_to_physical_constant = channel_group.attrs['voltToPhysicalConstant']
            
            # Get analog mask
            analog_mask = channel_group.attrs['analogMask']
            
            # Extract and scale data
            analog_data = self.file_handle[dataset_name] & analog_mask
            voltage_data = analog_data * bin_to_voltage_factor + bin_to_voltage_constant
            
            return voltage_data * volt_to_physical_factor + volt_to_physical_constant
            
        except KeyError as e:
            raise ValueError(f"Channel {channel}, block {block} not found in file: {e}")
    
    def get_channel_info(self, channel: int) -> Dict[str, Any]:
        """
        Get comprehensive channel information.
        
        Args:
            channel: Channel number
            
        Returns:
            Dictionary containing channel metadata
        """
        if self.file_handle is None:
            raise RuntimeError("File not opened. Use open() or context manager.")
            
        try:
            channel_group = self.file_handle[self._get_channel_group_name(channel)]
            
            return {
                'name': channel_group.attrs.get('name', f'Channel_{channel}').decode() if isinstance(channel_group.attrs.get('name', f'Channel_{channel}'), bytes) else channel_group.attrs.get('name', f'Channel_{channel}'),
                'physical_unit': channel_group.attrs.get('physicalUnit', '').decode() if isinstance(channel_group.attrs.get('physicalUnit', ''), bytes) else channel_group.attrs.get('physicalUnit', ''),
                'bin_to_volt_factor': channel_group.attrs.get('binToVoltFactor', 1.0),
                'bin_to_volt_constant': channel_group.attrs.get('binToVoltConstant', 0.0),
                'volt_to_physical_factor': channel_group.attrs.get('voltToPhysicalFactor', 1.0),
                'volt_to_physical_constant': channel_group.attrs.get('voltToPhysicalConstant', 0.0),
                'analog_mask': channel_group.attrs.get('analogMask', 0xFFFF),
                'marker_mask': channel_group.attrs.get('markerMask', 0x0000),
            }
            
        except KeyError as e:
            raise ValueError(f"Channel {channel} not found in file: {e}")
    
    def get_block_info(self, channel: int, block: int = 1) -> Dict[str, Any]:
        """
        Get block-specific information.
        
        Args:
            channel: Channel number
            block: Block number
            
        Returns:
            Dictionary containing block metadata
        """
        if self.file_handle is None:
            raise RuntimeError("File not opened. Use open() or context manager.")
            
        try:
            block_group = self.file_handle[self._get_block_name(channel, block)]
            
            return {
                'sample_rate_hz': block_group.attrs.get('sampleRateHertz', 1.0),
                'trigger_sample': block_group.attrs.get('triggerSample', 0),
                'trigger_time_seconds': block_group.attrs.get('triggerTimeSeconds', 0.0),
                'start_time': block_group.attrs.get('startTime', '').decode() if isinstance(block_group.attrs.get('startTime', ''), bytes) else block_group.attrs.get('startTime', ''),
            }
            
        except KeyError as e:
            raise ValueError(f"Channel {channel}, block {block} not found in file: {e}")
    
    def get_time_axis(self, channel: int, block: int = 1, time_scale: float = 1.0) -> np.ndarray:
        """
        Generate time axis for the data.
        
        Args:
            channel: Channel number
            block: Block number
            time_scale: Time scaling factor (1=seconds, 1000=milliseconds, 1000000=microseconds)
            
        Returns:
            Time axis array
        """
        block_info = self.get_block_info(channel, block)
        data_length = len(self.get_voltage_data(channel, block))
        
        sample_rate = block_info['sample_rate_hz']
        trigger_sample = block_info['trigger_sample']
        trigger_time = block_info['trigger_time_seconds']
        
        start_time = -trigger_sample / sample_rate * time_scale
        end_time = (data_length - trigger_sample) / sample_rate * time_scale
        
        t = np.arange(start_time, end_time, 1/sample_rate * time_scale)
        return t + trigger_time * time_scale
    
    def load_channel_data(self, channel: int, blocks: Optional[List[int]] = None, 
                         data_type: str = 'voltage') -> Dict[int, Dict[str, Any]]:
        """
        Load data from multiple blocks of a channel.
        
        Args:
            channel: Channel number
            blocks: List of block numbers (if None, loads all available blocks)
            data_type: Type of data to load ('voltage' or 'physical')
            
        Returns:
            Dictionary with block numbers as keys and data/metadata as values
        """
        if blocks is None:
            blocks = self.get_available_blocks(channel)
        
        if not blocks:
            raise ValueError(f"No blocks available for channel {channel}")
        
        channel_info = self.get_channel_info(channel)
        results = {}
        
        for block in blocks:
            try:
                block_info = self.get_block_info(channel, block)
                
                if data_type.lower() == 'voltage':
                    data = self.get_voltage_data(channel, block)
                    unit = 'V'
                elif data_type.lower() == 'physical':
                    data = self.get_physical_data(channel, block)
                    unit = channel_info['physical_unit']
                else:
                    raise ValueError("data_type must be 'voltage' or 'physical'")
                
                time_axis = self.get_time_axis(channel, block)
                
                results[block] = {
                    'data': data,
                    'time': time_axis,
                    'unit': unit,
                    'channel_info': channel_info,
                    'block_info': block_info,
                }
                
            except Exception as e:
                logging.warning(f"Failed to load channel {channel}, block {block}: {e}")
                continue
        
        return results
    
    def load_all_data(self, data_type: str = 'voltage') -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Load data from all available channels and blocks.
        
        Args:
            data_type: Type of data to load ('voltage' or 'physical')
            
        Returns:
            Nested dictionary: {channel: {block: {data, time, unit, info}}}
        """
        channels = self.get_available_channels()
        results = {}
        
        for channel in channels:
            try:
                results[channel] = self.load_channel_data(channel, data_type=data_type)
            except Exception as e:
                logging.warning(f"Failed to load channel {channel}: {e}")
                continue
        
        return results
    
    def get_file_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire TPC5 file.
        
        Returns:
            Dictionary containing file summary information
        """
        channels = self.get_available_channels()
        summary = {
            'filepath': str(self.filepath),
            'total_channels': len(channels),
            'channels': {}
        }
        
        for channel in channels:
            try:
                blocks = self.get_available_blocks(channel)
                channel_info = self.get_channel_info(channel)
                
                summary['channels'][channel] = {
                    'name': channel_info['name'],
                    'unit': channel_info['physical_unit'],
                    'total_blocks': len(blocks),
                    'blocks': blocks,
                }
                
                # Add info from first block
                if blocks:
                    block_info = self.get_block_info(channel, blocks[0])
                    summary['channels'][channel].update({
                        'sample_rate_hz': block_info['sample_rate_hz'],
                        'start_time': block_info['start_time'],
                    })
                    
            except Exception as e:
                logging.warning(f"Failed to get summary for channel {channel}: {e}")
                summary['channels'][channel] = {'error': str(e)}
        
        return summary


def load_tpc5_file(filepath: str, channels: Optional[List[int]] = None, 
                   blocks: Optional[List[int]] = None, 
                   data_type: str = 'voltage') -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Convenience function to quickly load TPC5 data.
    
    Args:
        filepath: Path to TPC5 file
        channels: List of channels to load (None for all)
        blocks: List of blocks to load (None for all)
        data_type: 'voltage' or 'physical'
        
    Returns:
        Nested dictionary with loaded data
    """
    with TPC5DataProcessor(filepath) as processor:
        if channels is None:
            return processor.load_all_data(data_type=data_type)
        
        results = {}
        for channel in channels:
            try:
                results[channel] = processor.load_channel_data(
                    channel, blocks=blocks, data_type=data_type
                )
            except Exception as e:
                logging.warning(f"Failed to load channel {channel}: {e}")
                
        return results


# Convenience functions for backward compatibility with original API
def get_voltage_data(file_handle, channel: int, block: int = 1) -> np.ndarray:
    """Legacy function for backward compatibility."""
    processor = TPC5DataProcessor.__new__(TPC5DataProcessor)
    processor.file_handle = file_handle
    return processor.get_voltage_data(channel, block)


def get_physical_data(file_handle, channel: int, block: int = 1) -> np.ndarray:
    """Legacy function for backward compatibility."""
    processor = TPC5DataProcessor.__new__(TPC5DataProcessor)
    processor.file_handle = file_handle
    return processor.get_physical_data(channel, block)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Using context manager (recommended)
    try:
        with TPC5DataProcessor("example.tpc5") as processor:
            # Get file summary
            summary = processor.get_file_summary()
            print("File Summary:")
            print(f"Total channels: {summary['total_channels']}")
            
            # Load data from specific channel
            if summary['total_channels'] > 0:
                channel = list(summary['channels'].keys())[0]
                data = processor.load_channel_data(channel, data_type='voltage')
                print(f"Loaded {len(data)} blocks from channel {channel}")
                
    except FileNotFoundError:
        print("Example file not found. This is expected in the example.")
    except Exception as e:
        print(f"Error: {e}")