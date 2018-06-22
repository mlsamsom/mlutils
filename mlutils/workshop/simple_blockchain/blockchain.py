# Module 1 create a blockchain

import datetime
from hashlib import sha256
import json
import requests
from urllib.parse import urlparse
from flask import Flask, jsonify

# Part 1 - Building a Blockchain
class BlockChain(object):
    """Block chain object
    """
    def __init__(self):
        # initialize chain
        self._chain = []

        # initialize mempool
        self.transactions = []

        # create genesis block
        self.create_block(proof=1, previous_hash='0')

        # create nodes
        self.nodes = set()

    @property
    def chain(self):
        return self._chain

    def create_block(self, proof, previous_hash):
        """Create a new block on the chain
        """
        block = {
            'index': len(self._chain) + 1,
            'timestamp': str( datetime.datetime.now() ),
            'proof': proof,
            'previous_hash': previous_hash,
            'transactions': self.transactions
        }
        self.transactions = []
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

    def is_chain_valid(self):
        """Perform validation checks
        """
        for i, block in enumerate(self.chain[1:], 1):
            previous_block = self.chain[i-1]
            # Check 1. hash chain
            if block['previous_hash'] != self.hash(previous_block):
                return False, "previous hash is not equal to hash {}".format(i)

            # Check 2. check if proof is valid
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = self._hash_op(previous_proof, proof)

            if hash_operation[:4] != '0000':
                return False, "hash {} is not valid".format(i)
        return True, "passed"

    def add_transaction(self, sender, reciever, amount):
        self.transactions.append(
            {'sender': sender,
             'reciever': reciever,
             'amount': amount}
        )
        previous_block = self.get_previous_block()
        return previous_block['index'] + 1

    def add_node(self, node_address):
        parsed_url = urlparse(node_address)
        self.nodes.add(parsed_url.netloc)

    def replace_chain(self):
        """Performs consensus protocol and addds a chain
        """
        # Find longest chain
        longest_chain = None
        max_length = len(self.chain)
        for node in self.nodes:
            response = requests.get("http://{}/get_chain".format(node))
            if response.status == 200:
                l = response.json()['length']
                chain = response.json()['chain']
                if l > max_length and self.is_chain_valid(chain):
                    max_length = l
                    longest_chain = chain
            if longest_chain:
                self.chain = longest_chain
                return True
            else:
                return False


# Part 2 - Mining out Blockchain

# create web app
app = Flask(__name__)

# create an address for node on Port 5000
node_address = str(uuid4().replace('-', ''))

# create blockchain
blockchain = BlockChain()

# mine a new block
@app.route('/mine_block', methods=['GET'])
def mine_block():
    # solve proof of work problem and get previous hash
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    #TODO decouple this
    blockchain.add_transaction(sender=node_address,
                               reciever='me',
                               amount=10)
    block = blockchain.create_block(proof, previous_hash)
    response = {
        'message': 'Congratulations you just mined a block',
        'index': block['index'],
        'timestamp': block['timestamp'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
        'transactions': block['transactions']
    }
    return jsonify(response), 200


# get full blockchain
@app.route('/get_chain', methods=['GET'])
def get_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain)
    }
    return jsonify(response), 200


#check if block chain is valid
@app.route('/is_valid', methods=['GET'])
def is_valid():
    ret, message = blockchain.is_chain_valid()
    if ret:
        response = {
            'message': 'Blockchain is valid',
        }
    else:
        response = {
            'message': 'Blockchain is not valid',
            'reason': message
        }

    return jsonify(response), 200


# add a block with transactions to the chain
@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    """Add a tranaction to a mined block
    """
    json_file = requests.get_json()
    transaction_keys = ['sender', 'reciever', 'amount']
    if not all(key in json_file for key in transaction_keys):
        return 'Some elements are missing', 400
    idx = blockchain.add_transaction(
        json_file['sender'],
        json_file['reciever'],
        json_file['amount'],
    )
    response = {'message': f'This transaction added to block {index}'}
    return jsonify(response), 201


# decentralize blockchain
@app.route('/connect_node', methods=['POST'])
def connect_node():
    """Connect a node to the blockchain network
    """
    json_file = requests.get_json()
    nodes =json_file.get('nodes')
    if nodes is None:
        return "no nodes", 400
    for node in nodes:
        blockchain.add_node(node)
    response = {'message': 'All nodes connected',
                'total_nodes': list(blockchain.nodes)}
    return jsonify(response), 201


# replace chain wiht longest chain
@app.route('/replace_chain', method=['GET'])
def replace_chain():
    is_chain_replaced = blockchain.is_chain_valid()
    if is_chain_replaced:
        response = {'message': 'nodes had difference chains, replaced by longest',
                    'new_chain': blockchain.chain}
    else:
        response = {'message': 'All good chain was longest',
                    'new_chain': blockchain.chain}
    return jsonify(response), 200


# run the app
app.run(host='0.0.0.0', port=5000)

