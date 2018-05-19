# Module 1 create a blockchain

import datetime
from hashlib import sha256
import json
from flask import Flask, jsonify

# Part 1 - Building a Blockchain
class BlockChain(object):
    """Block chain object
    """
    def __init__(self):
        # initialize chain
        self._chain = []

        # create genesis block
        self.create_block(proof=1, previous_hash='0')

    @property
    def chain(self):
        return self._chain

    def create_block(self, proof, previous_hash):
        """Create a new block on the chain
        """
        block = {
            'index': len(self._chain) + 1
            'timestamp': datetime.datetime.now(),
            'proof': proof,
            'previous_hash': previous_hash
        }
        self._chain.append(block)
        return block

    def get_previous_block(self):
        """Getter for last block on chain
        """
        return self._chain[-1]

    def _hash_op(self, previous_proof, new_proof):
        hshop = str(new_proof**2 - previous_proof**2)
        hash_operation = sha256(hshop.encode()).hexdigest()
        return hash_operation

    def proof_of_work(self, previous_proof):
        """Get the proof of work for a new block
        (Mining operation)
        """
        new_proof = 1
        check_proof = False

        while not check_proof:
            # perform hash operation
            hash_operation = self._hash_op(previous_proof, new_proof)

            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        """Verify blocks
        """
        # convert block to str
        encoded_block = json.dumps(block, sort_keys=True)
        encoded_block = encoded_block.encode()
        return sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        """Perform validation checks
        """
        for i, block in enumerate(chain[1:]):
            previous_block = chain[i-1]
            # Check 1. hash chain
            if block['previous_hash'] != self.hash(previous_block):
                return False

            # Check 2. check if proof is valid
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = self._hash_op(previous_proof, new_proof)

            if hash_operation[:4] != '0000':
                return False
        return True


# Part 2 - Mining out Blockchain
