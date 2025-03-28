"""
O1 Generated
Build me apython script that use the library skfolio and works as a interface where
 i can select which portfolio model i want to use and then add each type of portfolio
  arguments as pydantic objects so the  interfgace can be properly defined


"""


import sys
from typing import Optional
from pydantic import BaseModel, Field, validator

# Example skfolio portfolio classes:
# Adjust imports to the exact ones you use from skfolio
# e.g. from skfolio.portfolio import EqualWeighting, Markowitz, RiskBudgeting, ...
try:
    from skfolio.portfolio import Markowitz
    from skfolio.portfolio import EqualWeighting
    from skfolio.portfolio import HierarchicalRiskParity
except ImportError:
    print("Please install `skfolio` before running this script.")
    sys.exit(1)


# ------------------------------------------------------------------------------
# 1. Define Pydantic models for each Portfolio’s parameters
# ------------------------------------------------------------------------------

class EqualWeightingParams(BaseModel):
    """
    Example Pydantic model for EqualWeighting,
    which often does not have many user-tunable hyperparameters.
    If additional parameters exist, add them here.
    """
    # For demonstration, let's add a dummy parameter
    rebalance: bool = Field(
        default=True,
        description="Whether to rebalance at each period."
    )


class MarkowitzParams(BaseModel):
    """
    Example Pydantic model for Markowitz portfolio parameters.
    Adjust the fields to match the arguments your version of Markowitz uses.
    """
    risk_aversion: float = Field(
        default=1.0,
        description="Risk aversion parameter (usually > 0)."
    )
    allow_short: bool = Field(
        default=False,
        description="If True, short positions are allowed."
    )
    bounds: Optional[tuple[float, float]] = Field(
        default=(0.0, 1.0),
        description="Bounds on asset weights, e.g. (0, 1) for long-only."
    )

    @validator("risk_aversion")
    def risk_aversion_positive(cls, v):
        if v <= 0:
            raise ValueError("Risk aversion must be a positive float.")
        return v

    @validator("bounds")
    def bounds_valid(cls, v):
        lower, upper = v
        if lower > upper:
            raise ValueError("Lower bound cannot be greater than upper bound.")
        return v


class HierarchicalRiskParityParams(BaseModel):
    """
    Example Pydantic model for Hierarchical Risk Parity portfolio parameters.
    """
    linkage: str = Field(
        default="single",
        description="Linkage method for hierarchical clustering "
                    "(e.g. 'single', 'complete', 'ward', etc.)."
    )
    risk_measure: str = Field(
        default="variance",
        description="Risk measure used within the HRP algorithm "
                    "(e.g. 'variance', 'standard_deviation', etc.)."
    )


# ------------------------------------------------------------------------------
# 2. Create a mapping of portfolio “names” to (PydanticModel, skfolioClass)
# ------------------------------------------------------------------------------

PORTFOLIO_MODELS = {
    "equal_weighting": (EqualWeightingParams, EqualWeighting),
    "markowitz": (MarkowitzParams, Markowitz),
    "hrp": (HierarchicalRiskParityParams, HierarchicalRiskParity),
}


# ------------------------------------------------------------------------------
# 3. A helper function to prompt user for a portfolio and parameters
# ------------------------------------------------------------------------------

def select_portfolio_model() -> str:
    """
    Allows the user to pick a portfolio model from the available options.
    Returns the string key corresponding to the chosen portfolio type.
    """
    # Print available models
    print("\nAvailable Portfolio Models:")
    for i, model_name in enumerate(PORTFOLIO_MODELS.keys(), start=1):
        print(f"{i}. {model_name}")

    # Prompt user to select
    while True:
        choice = input("\nSelect a portfolio model by number or name: ").strip().lower()
        # If choice is a number, convert it to a name
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(PORTFOLIO_MODELS):
                return list(PORTFOLIO_MODELS.keys())[idx - 1]
        else:
            # If choice is a direct string match
            if choice in PORTFOLIO_MODELS:
                return choice

        print("Invalid selection. Please try again.")


def prompt_for_parameters(pydantic_model_class: BaseModel) -> BaseModel:
    """
    Dynamically prompt user for each field required by a given Pydantic model.
    Returns an instance of that model with validated parameters.
    """
    field_values = {}
    for field_name, field_def in pydantic_model_class.__fields__.items():
        field_type = field_def.type_
        field_default = field_def.default
        field_desc = field_def.field_info.description or ""

        # Prompt user
        prompt_text = f"{field_name} ({field_type.__name__}, default={field_default}): "
        if field_desc:
            prompt_text += f"[{field_desc}] "

        user_input = input(prompt_text).strip()

        # If the user just presses Enter, use the default
        if user_input == "":
            # Only use the default if it's not a Required field (no default).
            if field_default is not None:
                field_values[field_name] = field_default
                continue
            else:
                print(f"No default value for {field_name}, please provide a valid input.")
                return prompt_for_parameters(pydantic_model_class)

        # Try to convert input to the correct type
        try:
            # Simple type conversions for demonstration
            if field_type == bool:
                field_values[field_name] = user_input.lower() in ["true", "1", "yes", "y"]
            elif field_type == float:
                field_values[field_name] = float(user_input)
            elif field_type == int:
                field_values[field_name] = int(user_input)
            elif field_type == str:
                field_values[field_name] = user_input
            elif field_type == tuple:
                # For demonstration, assume a tuple of floats if the user inputs "0.0,1.0"
                splitted = user_input.split(",")
                field_values[field_name] = tuple(float(x.strip()) for x in splitted)
            else:
                # Fallback: attempt direct cast (rarely works for complex types)
                field_values[field_name] = field_type(user_input)
        except ValueError as ve:
            print(f"Invalid value for {field_name}: {ve}")
            return prompt_for_parameters(pydantic_model_class)

    # Now validate using Pydantic
    try:
        return pydantic_model_class(**field_values)
    except Exception as e:
        print(f"Error creating {pydantic_model_class.__name__}: {e}")
        print("Please re-enter the parameters.")
        return prompt_for_parameters(pydantic_model_class)


# ------------------------------------------------------------------------------
# 4. Main routine to put it all together
# ------------------------------------------------------------------------------

def main():
    """
    Main flow:
    1. The user selects the portfolio model.
    2. The user is prompted for each parameter in that model’s Pydantic schema.
    3. We instantiate the portfolio object from skfolio with the validated parameters.
    4. (Optionally) we do something with that portfolio, e.g., .fit(...)
    """
    print("\n=== Skfolio Portfolio Interface ===")

    # 1. User selects portfolio
    portfolio_key = select_portfolio_model()

    # 2. Retrieve the (Pydantic model, skfolio class)
    pydantic_cls, skfolio_cls = PORTFOLIO_MODELS[portfolio_key]

    # 3. Prompt the user for parameters for that model
    print(f"\nEnter parameters for {portfolio_key} portfolio:")
    pydantic_params = prompt_for_parameters(pydantic_cls)

    # 4. Instantiate the skfolio portfolio object
    #    (Here we show a simplified approach: pass Pydantic model fields as **kwargs)
    portfolio_instance = skfolio_cls(**pydantic_params.dict())
    print("\nPortfolio instance created successfully!")
    print("Parameters used:")
    print(pydantic_params.json(indent=4))

    # 5. Optionally, fit on data (dummy example below).
    #    Replace with real data and method calls relevant to your usage.
    """
    import numpy as np
    returns_data = np.random.randn(100, 5)  # e.g., 100 time periods, 5 assets
    portfolio_instance.fit(returns_data)
    weights = portfolio_instance.get_weights()
    print("Computed weights from fitted portfolio:", weights)
    """

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
