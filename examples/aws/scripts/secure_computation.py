#!/usr/bin/env python3

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from cryptography.fernet import Fernet
import json
import os
import socket
import threading
import queue

logger = logging.getLogger(__name__)

class SecureParty:
    """Secure computation party."""
    
    def __init__(self,
                 party_id: int,
                 num_parties: int,
                 port: int,
                 host: str = 'localhost'):
        """
        Initialize secure party.
        
        Args:
            party_id: Party identifier
            num_parties: Total number of parties
            port: Communication port
            host: Host address
        """
        self.party_id = party_id
        self.num_parties = num_parties
        self.port = port
        self.host = host
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize communication
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(num_parties - 1)
        
        # Message queues
        self.message_queues = {
            i: queue.Queue()
            for i in range(num_parties)
            if i != party_id
        }
        
    def start(self):
        """Start party communication."""
        try:
            # Start listening thread
            self.listening_thread = threading.Thread(
                target=self._listen_for_messages
            )
            self.listening_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting party: {str(e)}")
            raise
            
    def _listen_for_messages(self):
        """Listen for incoming messages."""
        try:
            while True:
                client_socket, address = self.socket.accept()
                
                # Receive message
                message = client_socket.recv(4096)
                
                # Decrypt message
                decrypted_message = self.cipher_suite.decrypt(message)
                message_data = json.loads(decrypted_message)
                
                # Add to appropriate queue
                sender_id = message_data['sender_id']
                self.message_queues[sender_id].put(message_data['data'])
                
                client_socket.close()
                
        except Exception as e:
            logger.error(f"Error listening for messages: {str(e)}")
            raise
            
    def send_message(self, 
                    receiver_id: int,
                    data: Dict) -> None:
        """
        Send message to another party.
        
        Args:
            receiver_id: Receiver party ID
            data: Message data
        """
        try:
            # Create message
            message = {
                'sender_id': self.party_id,
                'data': data
            }
            
            # Encrypt message
            encrypted_message = self.cipher_suite.encrypt(
                json.dumps(message).encode()
            )
            
            # Connect to receiver
            receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            receiver_socket.connect((self.host, self.port + receiver_id))
            
            # Send message
            receiver_socket.send(encrypted_message)
            receiver_socket.close()
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise
            
    def receive_message(self, 
                       sender_id: int,
                       timeout: Optional[float] = None) -> Dict:
        """
        Receive message from another party.
        
        Args:
            sender_id: Sender party ID
            timeout: Message timeout
            
        Returns:
            Received message data
        """
        try:
            return self.message_queues[sender_id].get(timeout=timeout)
            
        except queue.Empty:
            raise TimeoutError(f"Timeout waiting for message from party {sender_id}")
            
        except Exception as e:
            logger.error(f"Error receiving message: {str(e)}")
            raise
            
class SecureComputation:
    """Secure multi-party computation implementation."""
    
    def __init__(self,
                 party_id: int,
                 num_parties: int,
                 port: int,
                 host: str = 'localhost'):
        """
        Initialize secure computation.
        
        Args:
            party_id: Party identifier
            num_parties: Total number of parties
            port: Communication port
            host: Host address
        """
        self.party = SecureParty(party_id, num_parties, port, host)
        self.party.start()
        
    def secure_sum(self, 
                  values: np.ndarray,
                  timeout: Optional[float] = None) -> np.ndarray:
        """
        Compute secure sum of values.
        
        Args:
            values: Input values
            timeout: Computation timeout
            
        Returns:
            Secure sum result
        """
        try:
            # Split values into shares
            shares = self._split_into_shares(values)
            
            # Send shares to other parties
            for i in range(self.party.num_parties):
                if i != self.party.party_id:
                    self.party.send_message(i, {
                        'type': 'share',
                        'data': shares[i].tolist()
                    })
                    
            # Receive shares from other parties
            received_shares = []
            for i in range(self.party.num_parties):
                if i != self.party.party_id:
                    share_data = self.party.receive_message(i, timeout)
                    received_shares.append(np.array(share_data['data']))
                    
            # Compute sum
            result = shares[self.party.party_id]
            for share in received_shares:
                result += share
                
            return result
            
        except Exception as e:
            logger.error(f"Error in secure sum: {str(e)}")
            raise
            
    def _split_into_shares(self, values: np.ndarray) -> List[np.ndarray]:
        """Split values into secret shares."""
        try:
            shares = []
            for _ in range(self.party.num_parties - 1):
                # Generate random share
                share = np.random.rand(*values.shape)
                shares.append(share)
                
            # Compute last share
            last_share = values - sum(shares)
            shares.append(last_share)
            
            return shares
            
        except Exception as e:
            logger.error(f"Error splitting into shares: {str(e)}")
            raise
            
    def secure_mean(self,
                   values: np.ndarray,
                   timeout: Optional[float] = None) -> np.ndarray:
        """
        Compute secure mean of values.
        
        Args:
            values: Input values
            timeout: Computation timeout
            
        Returns:
            Secure mean result
        """
        try:
            # Compute secure sum
            sum_result = self.secure_sum(values, timeout)
            
            # Compute mean
            return sum_result / self.party.num_parties
            
        except Exception as e:
            logger.error(f"Error in secure mean: {str(e)}")
            raise
            
    def secure_variance(self,
                       values: np.ndarray,
                       timeout: Optional[float] = None) -> np.ndarray:
        """
        Compute secure variance of values.
        
        Args:
            values: Input values
            timeout: Computation timeout
            
        Returns:
            Secure variance result
        """
        try:
            # Compute secure mean
            mean_result = self.secure_mean(values, timeout)
            
            # Compute squared differences
            squared_diff = (values - mean_result) ** 2
            
            # Compute secure mean of squared differences
            return self.secure_mean(squared_diff, timeout)
            
        except Exception as e:
            logger.error(f"Error in secure variance: {str(e)}")
            raise
            
    def secure_covariance(self,
                         x: np.ndarray,
                         y: np.ndarray,
                         timeout: Optional[float] = None) -> np.ndarray:
        """
        Compute secure covariance between x and y.
        
        Args:
            x: First input array
            y: Second input array
            timeout: Computation timeout
            
        Returns:
            Secure covariance result
        """
        try:
            # Compute secure means
            x_mean = self.secure_mean(x, timeout)
            y_mean = self.secure_mean(y, timeout)
            
            # Compute product of differences
            diff_product = (x - x_mean) * (y - y_mean)
            
            # Compute secure mean of product
            return self.secure_mean(diff_product, timeout)
            
        except Exception as e:
            logger.error(f"Error in secure covariance: {str(e)}")
            raise 