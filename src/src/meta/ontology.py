# src/meta/ontology.py
"""
Lightweight System Ontology for New-Paradigm wrappers.
Provides ModuleDescriptor and SystemOntology singleton.
"""

import time
import threading
import json

class ModuleDescriptor:
    def __init__(self, name, role, inputs=None, outputs=None, invariants=None, version="v1"):
        self.name = name
        self.role = role
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.invariants = invariants or []
        self.version = version
        self.registered_at = time.time()

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "invariants": self.invariants,
            "version": self.version,
            "registered_at": self.registered_at
        }

class SystemOntology:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.modules = {}
        self.telemetry = []

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = SystemOntology()
            return cls._instance

    def register_module(self, descriptor):
        existing = self.modules.get(descriptor.name)
        if existing:
            # overwrite but log timestamp
            self.modules[descriptor.name] = descriptor
        else:
            self.modules[descriptor.name] = descriptor
        self.emit_telemetry({"event": "module_registered", "module": descriptor.name})

    def emit_telemetry(self, event):
        event["ts"] = time.time()
        self.telemetry.append(event)

    def export_manifest(self):
        return {n: d.to_dict() for n, d in self.modules.items()}

ontology = SystemOntology.instance()
