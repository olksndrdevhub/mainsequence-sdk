# src/instruments/base_instrument.py
from typing import Protocol, runtime_checkable, Optional
from pydantic import BaseModel, Field, PrivateAttr
from .json_codec import JSONMixin
import datetime
import json
class InstrumentModel(BaseModel, JSONMixin):
    """
    Common base for all Pydantic instrument models.
    Adds a shared optional 'main_sequence_uid' field and shared config.
    """
    main_sequence_asset_id :Optional[int] = Field(
        default=None,
        description="Optional UID linking this instrument to a main sequence record."
    )

    # Keep your existing behavior (QuantLib types, etc.)
    model_config = {"arbitrary_types_allowed": True}

    _valuation_date: Optional[datetime.datetime] =PrivateAttr(default=None)


    # public read access (still not serialized)
    @property
    def valuation_date(self) -> Optional[datetime.datetime]:
        return self._valuation_date

    # explicit setter method (per your request)
    def set_valuation_date(self, value: Optional[datetime.datetime]) -> None:
        self._valuation_date = value

    def serialize_for_backend(self):
        serialized={}
        data = self.model_dump_json()
        data = json.loads(data)
        serialized["instrument_type"] = type(self).__name__
        serialized["instrument"] = data

        return json.dumps(serialized)
